# Common-root planning diagnostic

**Twelve exposed root slots, including repeated startup states. No learned model, scientific qualification or architecture claim.**

Selection rules were fixed before execution and depend only on modeled scores, never native branch outcomes. Native cost is distance plus applied-command effort per action, averaged over four shared noise branches. Short plans hold their final command through step 24.

| Contrast | Mean baseline | Mean candidate | Improvement | Nonworse slots | Fixed rule |
|---|---:|---:|---:|---:|---|
| search | 0.189007 | 0.189041 | -0.018% | 10/12 | FAIL |
| horizon | 0.189041 | 0.175190 | +7.327% | 7/12 | FAIL |
| gain | 0.178382 | 0.173904 | +2.510% | 10/12 | FAIL |

The rule requires at least 3% and 0.001/action pooled improvement, no worse mean on each of the three source trajectories, and at least 8/12 nonworse slots. Each contrast stands alone.

| Source trajectory | Root | Short A | Short A+B | Long A | Union nominal | Union actual |
|---|---:|---:|---:|---:|---:|---:|
| nominal | 0 | 0.097709 | 0.096001 | 0.098092 | 0.095848 | 0.095848 |
| nominal | 50 | 0.210153 | 0.210153 | 0.209318 | 0.209318 | 0.209507 |
| nominal | 100 | 0.117422 | 0.133708 | 0.113834 | 0.117422 | 0.113834 |
| nominal | 150 | 0.217093 | 0.217093 | 0.219439 | 0.217093 | 0.217093 |
| public_gain | 0 | 0.097709 | 0.096001 | 0.098092 | 0.095848 | 0.095848 |
| public_gain | 50 | 0.340713 | 0.339414 | 0.271616 | 0.278519 | 0.271616 |
| public_gain | 100 | 0.130892 | 0.117926 | 0.094180 | 0.117676 | 0.094180 |
| public_gain | 150 | 0.217677 | 0.217677 | 0.220160 | 0.217677 | 0.217677 |
| true_state | 0 | 0.097709 | 0.096001 | 0.098092 | 0.095848 | 0.095848 |
| true_state | 50 | 0.328108 | 0.331615 | 0.275257 | 0.296851 | 0.275257 |
| true_state | 100 | 0.179861 | 0.179861 | 0.172903 | 0.171250 | 0.172903 |
| true_state | 150 | 0.233039 | 0.233039 | 0.231292 | 0.227231 | 0.227231 |

Stop this task/controller recipe as evidence for learned adaptation. No additional tuning sweep.

Execution 7.324s; independent replay 13.034s. Work: {"cross_score": 39648, "native_branch": 79296, "search_candidate": 294912, "selected_advance": 72}.

![All root costs and fixed contrasts](figure.png)

These are open-loop branches from reused states. The held-tail convention, privileged state/gain and small exposed root set limit the inference. H12 MPC would normally replan after acting, unlike these held-tail branches. The gain contrast ranks a common union populated by both gain-conditioned searches; it measures dynamics-dependent ranking, not a deployable online estimator or its compute value. No contrast establishes closed-loop improvement, generalization or a useful recurrent architecture. The minimum four-branch mean cost within the union is post-hoc descriptive context, never a deployable selected method. Slot timers include payload hashes but exclude their final completion write; the enclosing execution time includes it.
