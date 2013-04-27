__author__ = 'User'

from pyparsing import Optional, Literal, upcaseTokens

low_priority = Optional("LOW PRIORITY")
update_query = upcaseTokens(Literal('UPDATE')) +