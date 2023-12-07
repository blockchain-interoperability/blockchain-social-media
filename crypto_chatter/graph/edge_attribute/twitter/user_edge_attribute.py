from functools import partial
import numpy as np
from rich.progress import Progress

from crypto_chatter.data import CryptoChatterData
from crypto_chatter.utils import aggr_funcs
from crypto_chatter.utils.types import (
    EdgeList,
    NodeToIdMap,
    UserGraphEdgeAttributeKind,
    EdgeAttribute,
    UserInteractionKind,
    AttributeValues,
    AggrFunction,
)

def get_user_interaction(
    edges: EdgeList,
    node_to_ids: NodeToIdMap,
    data: CryptoChatterData,
    interaction_kind: UserInteractionKind,
    aggr_func: AggrFunction,
    progress: Progress | None = None,
) -> AttributeValues:
    if interaction_kind == "quote":
        interact_col = "quoted_status.user.id"
    elif interaction_kind == "reply":
        interact_col = "in_reply_to_user_id"
    else:
        raise NotImplementedError(f"User interaction kind {interaction_kind} not implemented")

    if interact_col not in data.columns:
        data.load([interact_col])

    # count instance where n2 is referred to by n1 in the interaction_kind
    if progress is not None:
        task = progress.add_task(
            f"getting {interaction_kind} count {aggr_func}",
            total=len(edges),
        )
    values = []
    # print(f"getting {interaction_kind} count")
    for n1, n2 in edges:
        # print(n1, n2)
        res = data.get(interact_col, node_to_ids[n1])
        interacted_users = res[~np.isnan(res)].astype(int)
        # print(f"to node interaction count: {len(interacted_users)}")
        aggr_interaction = aggr_funcs[aggr_func](interacted_users == n2)
        values += [aggr_interaction]
        if progress is not None:
            progress.advance(task)
    if progress is not None:
        progress.remove_task(task)

    return values

edge_attr_functions = {
    "total_quote_count": partial(get_user_interaction, interaction_kind="quote", aggr_func="sum"),
    "total_reply_count": partial(get_user_interaction, interaction_kind="reply", aggr_func="sum"),
    "avg_quote_count": partial(get_user_interaction, interaction_kind="quote", aggr_func="mean"),
    "avg_reply_count": partial(get_user_interaction, interaction_kind="reply", aggr_func="mean"),
}

def get_user_edge_attribute(
    edges: EdgeList,
    node_to_ids: NodeToIdMap,
    data: CryptoChatterData,
    kind: UserGraphEdgeAttributeKind,
    progress: Progress | None = None,
) -> EdgeAttribute:
    if kind in edge_attr_functions:
        values = edge_attr_functions[kind](
            edges=edges,
            data=data,
            node_to_ids=node_to_ids,
            progress=progress,
        )
        # edges need to be a list of tuple so they can be hased into a dict key
        return dict(zip(map(tuple, edges), values))
    else:
        raise ValueError(f"Unknown node attribute kind: {kind}")
