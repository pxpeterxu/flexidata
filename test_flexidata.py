__author__ = 'User'

from flexidata import *
import sqlparse
from sqlparse import sql as psql
from sqlparse import tokens as ptokens

stmt = sqlparse.parse("SELECT * FROM blah WHERE id = 1")[0]