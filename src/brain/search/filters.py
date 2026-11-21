"""The permission predicate that every index query carries in its WHERE clause.

Two parts, both required:
- ACL: the item's source ACL meets the principal set of every audience member. This is the
  ceiling. An empty ACL meets nothing, so an item with no ACL is visible to no one.
- Policy: the item's (label, tags) is one the business policy allows the asker to read. This
  only narrows below the ceiling.

Nothing in this module can widen access: an empty audience or an empty allow set is refused
or matches no rows.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from psycopg import sql

# One principal set per audience member. This unit asks for single-asker audiences only, but
# the predicate already requires visibility to every member.
Audience = Sequence[frozenset[str]]


@dataclass(frozen=True)
class PolicyFilter:
    """The (label, tags) combinations the policy allows this asker to read."""

    allowed_facts: frozenset[str]
    policy_version: str


def facts_key(label: str, tags: Iterable[str]) -> str:
    """The key the policy predicate matches an item on: its label and sorted tags."""
    return label + "|" + ",".join(sorted(tags))


def acl_visible(acl: Iterable[str], audience: Audience) -> bool:
    """The ACL check in Python, for re-deciding a result already returned."""
    principals = frozenset(acl)
    return bool(audience) and all(principals & member for member in audience)


def permission_clause(audience: Audience, policy: PolicyFilter, items: str = "i") -> sql.Composed:
    """The WHERE fragment for items aliased `items`: ACL for every member AND policy."""
    if not audience or any(not member for member in audience):
        raise ValueError("an audience needs at least one member, each with principals")
    alias = sql.Identifier(items)
    acl = [
        sql.SQL("{}.acl_principals && {}::text[]").format(alias, sql.Literal(sorted(member)))
        for member in audience
    ]
    policy_part = sql.SQL("{}.facts_key = ANY({}::text[])").format(
        alias, sql.Literal(sorted(policy.allowed_facts))
    )
    return sql.SQL("({})").format(sql.SQL(" AND ").join([*acl, policy_part]))
