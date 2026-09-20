"""Pure SGD categorical-state parsing with text features separate from labels."""

import hashlib
import json
import unicodedata
from dataclasses import asdict, dataclass

NOT_MENTIONED = "reserved:NOT_MENTIONED"
DONTCARE = "reserved:DONTCARE"
BINS = ("first_assignment", "revision", "assigned_retention", "unmentioned_retention", "clear")


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class Candidate:
    id: str
    text: str
    value: str | None


@dataclass(frozen=True)
class Query:
    query_id: str
    service: str
    slot: str
    service_description: str
    slot_description: str
    query_text: str
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True)
class Schema:
    queries: tuple[Query, ...]
    services: tuple[str, ...]
    service_slots: tuple[tuple[str, tuple[str, ...]], ...]

    def catalog(self):
        return [asdict(query) for query in self.queries]


def compile_schema(raw):
    require(isinstance(raw, list) and raw, "Schema must be a nonempty list")
    queries, services, all_slots = [], [], []
    for service in raw:
        name, description = service["service_name"], service["description"]
        require(isinstance(name, str) and name and name not in services, "Duplicate/invalid service")
        require(isinstance(description, str), "Service description")
        services.append(name)
        slots = []
        for slot in service["slots"]:
            slot_name, slot_description = slot["name"], slot["description"]
            require(isinstance(slot_name, str) and slot_name and slot_name not in slots,
                    f"Duplicate/invalid slot: {name}")
            require(isinstance(slot_description, str) and type(slot["is_categorical"]) is bool,
                    f"Slot schema: {name}/{slot_name}")
            slots.append(slot_name)
            if not slot["is_categorical"]:
                continue
            values = slot["possible_values"]
            require(isinstance(values, list) and values and all(isinstance(v, str) and v for v in values),
                    f"Categorical values: {name}/{slot_name}")
            require(len(values) == len(set(values)), f"Duplicate candidates: {name}/{slot_name}")
            require("dontcare" not in values, "Ambiguous literal dontcare ontology candidate")
            text = f"Service: {description}\nSlot: {slot_description}"
            candidates = [Candidate(NOT_MENTIONED, text + "\nValue: NOT_MENTIONED (no constraint stated)", None),
                          Candidate(DONTCARE, text + "\nValue: DONTCARE (no preference)", None)]
            candidates.extend(Candidate("value:" + value, text + "\nValue: " + value, value) for value in values)
            queries.append(Query(json.dumps([name, slot_name], separators=(",", ":"), ensure_ascii=False),
                                 name, slot_name, description, slot_description, text, tuple(candidates)))
        all_slots.append((name, tuple(slots)))
    return Schema(tuple(queries), tuple(services), tuple(all_slots))


def public_dialogue(raw):
    """Whitelist role/text only; no annotation or dialogue.services reaches features."""
    identifier = raw["dialogue_id"]
    require(isinstance(identifier, str) and identifier, "Dialogue identifier")
    require(isinstance(raw["turns"], list) and raw["turns"], "Empty dialogue")
    turns, user_turns = [], []
    for index, turn in enumerate(raw["turns"]):
        speaker, utterance = turn["speaker"], turn["utterance"]
        require(speaker in ("USER", "SYSTEM") and isinstance(utterance, str), "Public turn schema")
        turns.append({"speaker": speaker, "utterance": utterance})
        if speaker == "USER":
            previous = index - 1 if index and turns[index - 1]["speaker"] == "SYSTEM" else None
            user_turns.append({"turn_index": index, "previous_system_turn_index": previous})
    return {"dialogue_id": identifier, "turns": turns, "user_turns": user_turns}


def public_prefix(dialogue, turn_index):
    """The feature accessor for a prediction: no future turns or label metadata."""
    require(type(turn_index) is int and 0 <= turn_index < len(dialogue["turns"]), "Prefix index")
    require(dialogue["turns"][turn_index]["speaker"] == "USER", "Prediction requires USER turn")
    return [{"speaker": t["speaker"], "utterance": t["utterance"]}
            for t in dialogue["turns"][:turn_index + 1]]


def utterance_pair(dialogue, turn_index):
    prefix = public_prefix(dialogue, turn_index)
    previous = prefix[-2]["utterance"] if len(prefix) > 1 and prefix[-2]["speaker"] == "SYSTEM" else ""
    return {"previous_system": previous, "user": prefix[-1]["utterance"]}


def text_identity(dialogue):
    """NFKC, casefold and whitespace normalization; preserve ordered speaker roles."""
    normalized = [(t["speaker"], " ".join(unicodedata.normalize("NFKC", t["utterance"]).casefold().split()))
                  for t in dialogue["turns"]]
    payload = json.dumps(normalized, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def parse_dialogue(raw, schema, *, train_services):
    """Return public packet and separate evaluator labels; no persistent parser state."""
    public = public_dialogue(raw)
    by_service = {name: [] for name in schema.services}
    for query in schema.queries:
        by_service[query.service].append(query)
    all_slots = dict(schema.service_slots)
    previous, labels = {}, []
    for index, turn in enumerate(raw["turns"]):
        if turn["speaker"] != "USER":
            continue
        require(isinstance(turn["frames"], list), "USER frames must be a list")
        seen_services = set()
        for frame in turn["frames"]:
            service = frame["service"]
            require(service in by_service and service not in seen_services,
                    f"Unknown/duplicate USER service: {public['dialogue_id']}:{index}:{service}")
            seen_services.add(service)
            values = frame["state"]["slot_values"]
            require(isinstance(values, dict) and set(values) <= set(all_slots[service]),
                    f"State slot schema: {public['dialogue_id']}:{index}:{service}")
            for query in by_service[service]:
                value = values.get(query.slot)
                if query.slot not in values:
                    label_id, label_index = NOT_MENTIONED, 0
                else:
                    require(isinstance(value, list) and len(value) == 1 and isinstance(value[0], str),
                            f"Categorical singleton required: {public['dialogue_id']}:{index}:{service}/{query.slot}")
                    label_id = DONTCARE if value[0] == "dontcare" else "value:" + value[0]
                    ids = [candidate.id for candidate in query.candidates]
                    require(label_id in ids,
                            f"Out-of-schema category: {public['dialogue_id']}:{index}:{service}/{query.slot}:{value}")
                    label_index = ids.index(label_id)
                old = previous.get(query.query_id, NOT_MENTIONED)
                if label_id == old:
                    bin_name = "unmentioned_retention" if label_id == NOT_MENTIONED else "assigned_retention"
                elif old == NOT_MENTIONED:
                    bin_name = "first_assignment"
                elif label_id == NOT_MENTIONED:
                    bin_name = "clear"
                else:
                    bin_name = "revision"
                labels.append({"dialogue_id": public["dialogue_id"], "turn_index": index,
                               "service": service, "slot": query.slot, "query_id": query.query_id,
                               "label_id": label_id, "label_index": label_index, "bin": bin_name,
                               "is_dontcare": label_id == DONTCARE,
                               "unseen_service": service not in train_services})
                previous[query.query_id] = label_id
    return public, labels
