__author__ = 'User'

# Simple update parser
#
# Notes:
# - Table-names must be in the same database
# - Columns can only be specified for a single table
# - Both backticks and double-quotes are acceptable as field delimiters, but aren't escapable
# - No UTF8

from pyparsing import *
from select_parser import *

(UPDATE, SET) = map(CaselessKeyword, """UPDATE, SET""".replace(",", "").split())

keyword = MatchFirst(keyword.exprs + (UPDATE, SET))

identifier = ~keyword + Word(alphas, alphanums + "_")

quotable_identfier = QuotedString('`') \
                     | QuotedString('"', ) \
                     | identifier

set_clause = quotable_identfier + "=" + set_expr
set_clause_list = commaSeparatedList(set_clause)

update = (UPDATE + quotable_identfier('table') + SET + set_clause_list('set_clause_list')