"""Sparse exact action effects for relation transport: untrained prototype.

Native chess computes candidate graph changes in a fixed root-player frame.
For row-stochastic child operators P+D, propagation is pP+pD. This linear
identity is established mathematics, not a new theorem. It avoids storing a
full floating-point transition matrix for every candidate during propagation.
No learned model, speed improvement or chess-quality benefit is asserted here.
"""
import chess
import numpy as np
import torch
from openjev.research.chess_transport import transitions

VERSION='counterfactual-attack-operator-delta-v1'


def fixed_frame_relations(board,perspective):
    if type(board) is not chess.Board or board.chess960 or not board.is_valid():
        raise ValueError('Expected a valid standard board')
    if type(perspective) is not bool:raise ValueError('Expected a player color')
    canonical=lambda s:s if perspective else chess.square_mirror(s)
    result=np.zeros((2,64,64),dtype=np.uint8)
    for source,piece in board.piece_map().items():
        color=int(piece.color!=perspective)
        for target in board.attacks(source):result[color,canonical(source),canonical(target)]=1
    return result


def pack(boards,*,dtype=torch.float32):
    """Construct root operators and nonzero normalized child-minus-root entries.

Normalization and empty-row self-loops are included in the difference. Deleting
an edge can change every coefficient in that source row. Histories and input
boards are preserved; no teacher labels enter this function. Candidate order is
lexicographic UCI and includes underpromotion. Padded slots use the root operator.
"""
    boards=list(boards)
    if not boards or dtype not in (torch.float32,torch.float64):raise ValueError('Expected boards and floating dtype')
    roots=[];changes=[];values=[];menus=[];child_count=0;changed_rows=0
    for b,board in enumerate(boards):
        raw=fixed_frame_relations(board,board.turn)
        names=sorted(move.uci() for move in board.legal_moves)
        if not names or board.is_game_over(claim_draw=False):raise ValueError('Expected nonterminal legal candidates')
        menus.append(names);root=transitions(torch.from_numpy(raw)[None],dtype)[0];roots.append(root)
        for m,uci in enumerate(names):
            child=board.copy(stack=True);child.push_uci(uci)
            child_raw=fixed_frame_relations(child,board.turn)
            operator=transitions(torch.from_numpy(child_raw)[None],dtype)[0]
            delta=operator-root;where=delta.nonzero();changed_rows+=int(delta.ne(0).any(-1).sum())
            prefix=torch.tensor([b,m],dtype=torch.long).expand(len(where),2)
            changes.append(torch.cat((prefix,where),-1));values.append(delta[where[:,0],where[:,1],where[:,2]])
            child_count+=1
    width=max(map(len,menus));mask=torch.zeros(len(boards),width,dtype=torch.bool)
    for b,names in enumerate(menus):mask[b,:len(names)]=True
    root=torch.stack(roots);indices=torch.cat(changes);delta_values=torch.cat(values)
    return {'roots':root,'indices':indices,'values':delta_values,'mask':mask,'menus':menus,
            'counts':{'roots':len(boards),'legal_children':child_count,'changed_normalized_entries':len(delta_values),
                      'changed_relation_rows':changed_rows,'sparse_tensor_bytes':sum(t.numel()*t.element_size() for t in (root,indices,delta_values)),
                      'dense_valid_child_operator_bytes':child_count*4*64*64*root.element_size()}}


def propagate(probabilities,gates,roots,indices,values):
    """Apply each candidate's four operators to two flows and mix by gates.

Probabilities: [batch,moves,flows,64]; gates: [batch,moves,flows,4].
Indices: [entries,5] for batch/move/relation/source/destination. Each normalized
entry is listed at most once by pack(). Values are fixed graph data; gradients
through probabilities and gates are preserved. No candidate-dense graph tensor
is materialized. Padding is a caller concern and receives root-only transport.
"""
    if probabilities.ndim!=4:raise ValueError('Expected four probability dimensions')
    batch,moves,flows,nodes=probabilities.shape
    if min(batch,moves,flows)<1 or nodes!=64 or roots.shape!=(batch,4,64,64) or gates.shape!=(batch,moves,flows,4):
        raise ValueError('Invalid propagation shape')
    if indices.ndim!=2 or indices.shape[1]!=5 or indices.dtype!=torch.long or values.shape!=(len(indices),):
        raise ValueError('Invalid sparse delta')
    if any(t.device!=probabilities.device for t in (gates,roots,indices,values)):
        raise ValueError('Device mismatch')
    if any(t.dtype!=probabilities.dtype for t in (gates,roots,values)):
        raise ValueError('Dtype mismatch')
    if not all(torch.isfinite(t).all() for t in (probabilities,gates,roots,values)):
        raise ValueError('Nonfinite propagation input')
    bounds=torch.tensor([batch,moves,4,64,64],device=indices.device)
    if ((indices<0)|(indices>=bounds)).any():raise ValueError('Sparse delta index out of bounds')
    base=torch.einsum('blfn,brnm->blfrm',probabilities,roots)
    output=(base*gates[...,None]).sum(-2)
    b,m,r,s,t=indices.unbind(-1)
    contribution=probabilities[b,m,:,s]*gates[b,m,:,r]*values[:,None]
    flow=torch.arange(flows,device=indices.device)[None]
    destinations=(((b[:,None]*moves+m[:,None])*flows+flow)*64+t[:,None]).reshape(-1)
    return output.reshape(-1).index_add(0,destinations,contribution.reshape(-1)).reshape(batch,moves,flows,64)
