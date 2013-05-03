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

class From(psql.TokenList):
    """A FROM clause for SELECTs."""
    __slots__ = ('value', 'ttype', 'tokens')

Group = ptokens.Group;

# These are actually weird ways of instantiating new types
Group.OrderBy = ptokens.Group.OrderBy;
Group.GroupBy = ptokens.Group.GroupBy;
Group.Having = ptokens.Group.Having;
Group.From = ptokens.Group.From;

def group_list(tlist, starting_keyword, instance):
    """
    Group tokens together based on the starting keyword (e.g. ORDER, WHERE, etc.)
    :param tlist: parsed token list
    :param starting_keyword: keyword to trigger formation of group
    :type starting_keyword: str
    :param instance: the Class to make the group with
    """
    [group_list(sgroup, starting_keyword, instance) for sgroup in tlist.get_sublists()
     if not isinstance(sgroup, instance)]
    idx = 0
    token = tlist.token_next_match(idx, ptokens.Keyword, starting_keyword)
    stopwords = ['FROM', 'WHERE', 'ORDER', 'GROUP', 'LIMIT', 'UNION', 'HAVING']
    while token:
        tidx = tlist.token_index(token)
        end = tlist.token_next_match(tidx + 1, ptokens.Keyword, stopwords)
        end2 = tlist.token_next_by_instance(tidx + 1, psql.Where)

        index = 99999 if end is None else tlist.token_index(end)
        index2 = 99999 if end2 is None else tlist.token_index(end2)
        first = min(index, index2)
        end = tlist.tokens[first] if first != 99999 else None

        if end is None:
            end = tlist._groupable_tokens[-1]
        else:
            end = tlist.tokens[tlist.token_index(end) - 1]
        group = tlist.group_tokens(instance,
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

def group_from(tlist):
    group_list(tlist, 'FROM', From)

def parse(sql):
    parsed = sqlparse.parse(sql)
    group_from(parsed[0])
    group_group_by(parsed[0])
    group_having(parsed[0])
    group_order_by(parsed[0])

    return parsed
