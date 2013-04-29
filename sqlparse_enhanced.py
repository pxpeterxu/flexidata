__author__ = 'User'

import sqlparse
from sqlparse import sql as psql
from sqlparse import tokens as ptokens

#
# Extensions to sqlparse
#
class OrderBy(psql.TokenList):
    """A ORDER BY clause."""
    __slots__ = ('value', 'ttype', 'tokens')

class GroupBy(psql.TokenList):
    """A GROUP BY clause."""
    __slots__ = ('value', 'ttype', 'tokens')

class Having(psql.TokenList):
    """A HAVING clause."""
    __slots__ = ('value', 'ttype', 'tokens')

Group = ptokens.Group;

# These are actually weird ways of instantiating new types
Group.OrderBy = ptokens.Group.OrderBy;
Group.GroupBy = ptokens.Group.GroupBy;
Group.Having = ptokens.Group.Having;

def group_list(tlist, starting_keyword, group_instance):
    """
    Group tokens together based on the starting keyword (e.g. ORDER, WHERE, etc.)
    :param tlist: parsed token list
    :param starting_keyword: keyword to trigger formation of group
    :type starting_keyword: str
    :param group_instance: the Class to make the group with
    """
    [group_list(sgroup, starting_keyword, group_instance) for sgroup in tlist.get_sublists()
     if not isinstance(sgroup, group_instance)]
    idx = 0
    token = tlist.token_next_match(idx, ptokens.Keyword, starting_keyword)
    stopwords = ('WHERE', 'ORDER', 'GROUP', 'LIMIT', 'UNION', 'HAVING')
    while token:
        tidx = tlist.token_index(token)
        end = tlist.token_next_match(tidx + 1, ptokens.Keyword, stopwords)
        if end is None:
            end = tlist._groupable_tokens[-1]
        else:
            end = tlist.tokens[tlist.token_index(end) - 1]
        group = tlist.group_tokens(group_instance,
            tlist.tokens_between(token, end),
            ignore_ws=True)
        idx = tlist.token_index(group)
        token = tlist.token_next_match(idx, ptokens.Keyword, starting_keyword)

def group_order_by(tlist):
    group_list(tlist, 'ORDER', OrderBy)

def group_group_by(tlist):
    group_list(tlist, 'GROUP', GroupBy)

def group_having(tlist):
    group_list(tlist, 'HAVING', Having)

def parse(sql):
    parsed = sqlparse.parse(sql)
    group_order_by(parsed[0])
    group_group_by(parsed[0])
    group_having(parsed[0])

    return parsed
