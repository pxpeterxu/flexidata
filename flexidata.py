__author__ = 'User'

import sqlparse

from sqlparse import sql as psql
from sqlparse import tokens as ptokens

import sqlparse_enhanced as psqle

import pymysql
import re
import itertools

from collections import defaultdict, OrderedDict
import copy
import settings

original_conn = pymysql.connect(
    db=settings.flexidata_database,
    user=settings.flexidata_username,
    passwd=settings.flexidata_password,
    host=settings.flexidata_host)


class Connection(object):
    """
    Wraps around a DBAPI 2.0 Connection
    """

    def __init__(self, conn):
        """
        :type conn: pymysql.connections.Connection
        """
        self.conn = conn
        self._refresh_schemas()

    def begin(self):
        self.conn.begin()
        self._refresh_schemas()

    def close(self):
        self.conn.close()

    def commit(self):
        self.conn.commit()
        # New transaction, so refresh schema
        self._refresh_schemas()

    def rollback(self):
        self.conn.rollback()

    def cursor(self):
        return Cursor(self.conn.cursor(), self)

    def escape(self, obj):
        self.conn.escape(obj)

    def _refresh_schemas(self):
        print 'refreshing'
        self.schemas, self.num_rows, primary_keys = retrieve_schemas(self.conn)
        self.primary_keys = {}
        for table_name, key in primary_keys.iteritems():
            if '__' in table_name:
                base_table_name = table_name[:table_name.find('__')]
                if not base_table_name in self.primary_keys:
                    self.primary_keys[base_table_name] = key

        self.schemas = group_schemas(self.schemas, self.num_rows)
        print 'refreshed'
        # self.schemas is in the form of
        # 'table_name' => [(subtable_number, subtable_schema, num_rows)]


class Cursor(object):
    """
    Wraps around a DBAPI 2.0 Cursor
    """

    def __init__(self, cursor, conn):
        """
        :type cursor: pymysql.cursors.Cursor
        :type conn: Connection
        """
        self.cursor = cursor
        self.conn = conn

    @property
    def description(self):
        return self.cursor.description

    @property
    def rowcount(self):
        return self.cursor.rowcount

    def callproc(self, procname, args=()):
        return self.cursor.callproc(procname, args)

    def close(self):
        self.cursor.close()

    def _prepare_schema(self, table_name, query_schema):
        schemas = self.conn.schemas

        # Finds existing table schema
        if table_name not in schemas:
            original_schema = {}
            old_table_version = None
            new_table_version = -1
        else:
            original_schema = schemas[table_name][0][1]
            old_table_version = schemas[table_name][0][0]
            new_table_version = old_table_version + 1

        # Check what we need to change
        create_schema, add_schema, modify_schema = generate_new_schema(
            original_schema, query_schema)

        # If we need to change the schema
        if add_schema or modify_schema:
            # TODO(hsource) Currently all and any schema changes result in a new table; wasteful
            real_table_name = make_real_table_name(table_name, new_table_version)
            create_table_sql = generate_create_table(real_table_name, create_schema)
            self.cursor.execute(create_table_sql)

            # Triggers
            if old_table_version is not None:
                old_table_name = make_real_table_name(table_name, old_table_version)
                shared_columns = [column for column in create_schema
                                  if column not in add_schema and column not in modify_schema]
                create_trigger_sql = generate_triggers(old_table_name, real_table_name,
                                                       shared_columns)
                self.cursor.execute(create_trigger_sql)

            self.conn._refresh_schemas()
        else:
            real_table_name = make_real_table_name(table_name, schemas[table_name][0][0])

        return real_table_name


    def _prepare_for_insert(self, stmt):
        """
        Prepare for an insert
        :param stmt:
        :return:
        """
        table_name, columns = insert_find_table_info(stmt)
        value_sets = insert_find_values(stmt)
        types, lengths = find_minimum_types_for_values(value_sets)

        # Finds query schema
        query_minimum_schema = OrderedDict()
        for info in zip(columns, types, lengths):
            query_minimum_schema[info[0]] = {'type': info[1], 'length': info[2]}

        return self._prepare_schema(table_name, query_minimum_schema)

    def _prepare_for_update(self, stmt):
        table_name, row_data, _ = update_find_data(stmt)
        types, lengths = find_minimum_types_for_values([row_data.values()])

        query_minimum_schema = OrderedDict()
        for info in zip(row_data.keys(), types, lengths):
            query_minimum_schema[info[0]] = {'type': info[1], 'length': info[2]}

        return self._prepare_schema(table_name, query_minimum_schema)

    def _prepare_for_delete(self, stmt):
        table_name, row_data, _ = update_find_data(stmt)
        types, lengths = find_minimum_types_for_values([row_data.values()])

        query_minimum_schema = OrderedDict()
        for info in zip(row_data.keys(), types, lengths):
            query_minimum_schema[info[0]] = {'type': info[1], 'length': info[2]}

        return self._prepare_schema(table_name, query_minimum_schema)

    def _insert_data_for_update(self, stmt, real_table_name):
        """
        Propagates data matching the UPDATE conditions from the older tables to the newer
        one.
        """
        where_token = stmt.token_next_by_instance(0, psql.Where)
        table_name, _, _ = update_find_data(stmt)

        propagate_sql = generate_propagate_sql(real_table_name, table_name, self.conn.schemas[real_table_name], 'sid', where_token)
        self.cursor.execute('SET @disable_triggers = 1;')
        self.cursor.execute(propagate_sql)
        self.cursor.execute('SET @disable_triggers = NULL;')

    def _prepare_for_select(self, stmt):
        # Get basic information
        tables = select_find_tables(stmt)
        if len(tables) == 1:
            (default_table,) = tables
        else:
            default_table = None

        # Get columns in JOIN conditions
        #[FROM table_references
        #[WHERE where_condition]
        #[GROUP BY {col_name | expr | position}
        #          [ASC | DESC], ... [WITH ROLLUP]]
        #[HAVING where_condition]
        #[ORDER BY {col_name | expr | position}
        #          [ASC | DESC], ...]
        select_columns = select_find_columns(stmt, default_table)
        where_columns = find_columns_in_where(stmt, default_table)
        having_columns = find_columns_in_having(stmt, default_table)
        order_by_columns = find_columns_in_orderby(stmt, default_table)
        group_by_columns = find_columns_in_groupby(stmt, default_table)

        columns = select_columns | where_columns | having_columns | order_by_columns | group_by_columns

        # Get tables we used
        used_tables = set(table for table, column in columns) | tables
        schemas = self.conn.schemas

        # Get the identifiers in SELECT [] FROM ... to be replaced
        from_token = stmt.token_next_by_instance(0, psqle.From)
        column_tokens = stmt.tokens_between(stmt.tokens[0], from_token, exclude_end=True)
        group_by = stmt.token_next_by_instance(0, psqle.GroupBy)
        order_by = stmt.token_next_by_instance(0, psqle.OrderBy)
        where = stmt.token_next_by_instance(0, psql.Where)
        having = stmt.token_next_by_instance(0, psqle.Having)

        table_strings = {}
        for table in tables:
            schema = schemas[table]
            column_info = schema[0][1]
            if (None, '*') in columns or (table, '*') in columns:
                columns_to_get = None # Get all columns
            else:
                columns_to_get = columns


            first_table, column_versions, join_tables = generate_select_arguments(
                table, schema, columns_to_get)
            table_strings[table] = make_join_string(first_table, join_tables, column_info.keys()[0])
            replace_identifiers(column_tokens, {table: column_versions})
            if group_by is not None:
                replace_identifiers(group_by.tokens, {table: column_versions})
            if order_by is not None:
                replace_identifiers(order_by.tokens, {table: column_versions})
            if having is not None:
                replace_where_identifiers(having, {table: column_versions})
            if where is not None:
                replace_where_identifiers(where, {table: column_versions})

        # TODO(peterxu) make it so that we replace table names instead of just replacing everything
        from_token.tokens = psqle.parse('FROM ' + ', '.join(table_strings.itervalues()))
        return stmt

    def execute(self, query, args=None):
        if args is not None:
            query = query % self.conn.escape(args)
        stmts = psqle.parse(query)

        for stmt in stmts:
            if stmt.token_next_match(0, ptokens.DML, 'INSERT') is not None:
                real_table_name = self._prepare_for_insert(stmt)
                insert_replace_table_name(stmt, real_table_name)
            if stmt.token_next_match(0, ptokens.DML, 'UPDATE') is not None:
                real_table_name = self._prepare_for_update(stmt)
                self._insert_data_for_update(stmt, real_table_name)
                update_replace_table_name(stmt, real_table_name)
            if stmt.token_next_match(0, ptokens.DML, 'DELETE') is not None:
                real_table_name = self._prepare_for_delete(stmt)
                update_replace_table_name(stmt, real_table_name)

            if stmt.token_next_match(0, ptokens.DML, 'SELECT') is not None:
                # Use advanced query parser
                self._prepare_for_select(stmt)

        stmts = [str(stmt) for stmt in stmts]
        query = ';\n'.join(stmts)
        return self.cursor.execute(query)

    def executemany(self, query, args):
        return self.cursor.executemany(query, args)

    def fetchone(self):
        return self.cursor.fetchone()

    def fetchmany(self, size=None):
        return self.cursor.fetchmany(size)

    def fetchall(self):
        return self.cursor.fetchall()

    def nextset(self):
        return self.cursor.nextset()

    @property
    def arraysize(self):
        return self.cursor.arraysize

    @arraysize.setter
    def arraysize(self, value):
        self.cursor.arraysize = value

    def setinputsizes(self, sizes):
        self.cursor.setinputsizes(sizes)

    def setoutputsize(self, size, column=None):
        self.cursor.setoutputsizes(size, column)


def find_tokens_until_match(token, until_token_filter):
    """
    Find all the tokens starting from token and ending before a token that
    matches one of the token_specs
    :param token: token to start searching at
    :type token: sqlparse.sql.Token
    :param until_token_filter: conditions on the kind of token to stop at. The function
                              should return True on match, False otherwise
    :type until_token_filter: function
    :return: all tokens (including 'token') before a token that matches until_token_specs
    """
    cur = token
    while cur is not None:
        if until_token_filter(cur):
            break
        cur = cur.token_next()

    if cur is None:
        return


def find_tokens_by_instance(tokens, token_class, recursive=False):
    """
    Utility function to find all tokens in a list of tokens that have a given Python type

    :type tokens: list of sqlparse.sql.Token
    :type token_class: sqlparse.sql.Token
    :rtype list of sqlparse.sql.Token
    """
    tokens_found = []
    for token in tokens:
        if isinstance(token, token_class):
            tokens_found.append(token)
        elif recursive and isinstance(token, psql.TokenList):
            tokens_found += find_tokens_by_instance(token.tokens, token_class, recursive)
    return tokens_found


def find_tokens_by_type(tokens, token_type, recursive=False):
    """
    Utility function to find all tokens in a list of tokens with given token type

    :type tokens: list of sqlparse.sql.Token
    :type token_type: sqlparse.tokens._TokenType or iterable
    :rtype list of sqlparse.sql.Token
    """
    tokens_found = []
    for token in tokens:
        if token.ttype is not None and token_type in token.ttype.split():
            tokens_found.append(token)
        elif recursive and isinstance(token, psql.TokenList):
            tokens_found += find_tokens_by_type(token.tokens, token_type, recursive)
    return tokens_found


def separate_table_and_column(identifier, default_table):
    """
    Separate the table and column in a column identifier (e.g. table.column, or just `column`)
    :param identifier: Identifier token
    :type identifier: sqlparse.sql.Identifier
    :return (table, column) tuple
    """
    # Possible schemes
    # table.column AS alias: parsed as
    #   table (Name)
    #   column (Name)
    #   AS (Keyword)
    #   alias (Identifier)
    #       alias (Name)
    # In all cases, the last Name in the identifier is the column

    name_tokens = find_tokens_by_type(identifier.tokens, ptokens.Name)
    names = [str(token).strip('"`') for token in name_tokens]
    table = default_table if len(names) == 1 else names[0]
    column = names[-1]

    # For SELECT *; don't override SELECT table1.*
    if column == '*' and len(name_tokens) == 1:
        table = None

    return table, column


def find_columns_in_condition(stmt, default_table):
    """
    Find all column names in a WHERE or HAVING statement
    :param stmt: WHERE or HAVING statement
    :type stmt: sqlparse.sql.TokenList
    :return: a set of [(table, column), ...] as strings
    """
    comparisons = find_tokens_by_instance(stmt.tokens, psql.Comparison, True)

    columns = set()
    # For each comparison
    for comparison in comparisons:
        identifier = comparison.token_next_by_instance(0, psql.Identifier)
        if identifier is not None:
            columns.add(separate_table_and_column(identifier, default_table))

    return columns


def find_columns_in_where(stmt, default_table):
    """
    Find all column names in a WHERE statement
    :param stmt: top-level parsed SQL statement
    :param default_table: default table to assume for unqualified columns
    :return: a set of [(table, column), ...] as strings
    """
    where = stmt.token_next_by_instance(0, psql.Where)
    if where is None:
        return set()
    return find_columns_in_condition(where, default_table)


def find_columns_in_having(stmt, default_table):
    """
    Find all column names in a HAVING statement
    :param stmt: top-level parsed SQL statement
    :param default_table: default table to assume for unqualified columns
    :return: a set of [(table, column), ...] as strings
    """
    having = stmt.token_next_by_instance(0, psqle.Having)
    if having is None:
        return set()
    return find_columns_in_condition(having, default_table)


def find_columns_in_orderby(stmt, default_table):
    """
    Find all column names in an ORDER BY
    :param stmt: top-level parsed SQL statement
    :param default_table: default table to assume for unqualified columns
    :return: a set of [(table, column), ...] as strings
    """
    order_by = stmt.token_next_by_instance(0, psqle.OrderBy)
    if order_by is None:
        return set()
    return find_columns_in_orderby_groupby(order_by, default_table)


def find_columns_in_groupby(stmt, default_table):
    """
    Find all column names in an GROUP BY
    :param stmt: top-level parsed SQL statement
    :param default_table: default table to assume for unqualified columns
    :return: a set of [(table, column), ...] as strings
    """
    group_by = stmt.token_next_by_instance(0, psqle.GroupBy)
    if group_by is None:
        return set()
    return find_columns_in_orderby_groupby(group_by, default_table)


def find_columns_in_orderby_groupby(stmt, default_table):
    """
    Find all column names in an ORDER BY or GROUP BY statement
    :param stmt: ORDER BY or GROUP BY statement
    :type stmt: sqlparse.sql.TokenList
    :return: a set of [(table, column), ...] as strings
    """
    identifiers = find_tokens_by_instance(stmt.tokens, psql.Identifier)
    return set(separate_table_and_column(id, default_table) for id in identifiers)


def find_identifiers_with_name_sub_token(tokens):
    """
    Find all column/table-name identifiers in the list of tokens given.
    :type tokens: list of Token
    :rtype: list of Identifier
    """
    found_identifiers = []
    for token in tokens:
        if (isinstance(token, psql.Identifier) and
                token.token_next_by_type(0, ptokens.Name) is not None):
            found_identifiers.append(token)
        elif isinstance(token, psql.TokenList):
            found_identifiers.extend(find_identifiers_with_name_sub_token(token.tokens))
    return found_identifiers


def find_comparison_with_identifier(token, identifier):
    """
    Recursively looks for a Comparison object that contains the specified identifier inside.
    """
    for sub_token in token.tokens:
        if isinstance(sub_token, psql.Comparison):
            identifiers = find_tokens_by_instance(sub_token.tokens, psql.Identifier, True)
            if identifier in identifiers:
                return sub_token, token
        elif isinstance(sub_token, psql.TokenList):
            sub_return = find_comparison_with_identifier(sub_token, identifier)
            if sub_return is not None:
                return sub_token, token

    return None


def find_columns_in_where_old(stmt):
    """
    Finds all column names in a statement (for seeing if schema change is necessary).

    :type stmt: sqlparse.sql.Statement
    """
    columns = set()
    where = stmt.token_next_by_instance(0, psql.Where)
    assert isinstance(where, psql.Where)
    comparisons = find_tokens_by_instance(where.tokens, psql.Comparison, True)

    for comparison in comparisons:
        assert isinstance(comparison, psql.Comparison)
        identifier = comparison.token_next_by_instance(0, psql.Identifier)
        names = find_tokens_by_type(identifier.tokens, ptokens.Name)
        names = [str(name).strip('"`') for name in names]
        assert 1 <= len(names) <= 2
        if len(names) == 2:
            columns.add((names[0], names[1]))
        else:
            columns.add((names[0],))

    return columns


def print_token_children(root_token, tabs=0):
    """
    For debugging purposes, use this on a sqlparsing parsed Statement to print out the whole tree.

    :type root_token: sqlparse.sql.TokenList
    """
    if not root_token.is_group():
        return

    for idx, token in enumerate(root_token.tokens):
        print(' ' * (tabs * 2) + '{}. {} - {}, {}'.format(idx, repr(token), type(token),
            token.ttype))
        print_token_children(token, tabs + 1)


def insert_find_table_info_tokens(stmt):
    """
    Given an INSERT query, find tokens that relate to the table information
    :param stmt: parsed statement tree from sqlparse
    :type stmt: sqlparse.sql.TokenList
    :return: the token for the name, and the tokens for the column names of the table
    """
    query_type_token = stmt.token_next_by_type(0, ptokens.DML)
    search_start_index = stmt.token_index(query_type_token) + 1

    # The parser sucks so we have to take care of two cases; grr should've learned
    # to write my own parser
    function = stmt.token_next_by_instance(search_start_index, psql.Function)
    identifier = stmt.token_next_by_instance(search_start_index, psql.Identifier)

    # If there's no function, or the first identifier comes before the first function
    if function is None or (identifier is not None
                            and stmt.token_index(identifier) < stmt.token_index(function)):
        parenthesis = function.token_next_by_instance(stmt.token_index(identifier) + 1,
            psql.Parenthesis)
    else:  # We have a function
        identifier = function.token_next_by_instance(0, psql.Identifier)
        parenthesis = function.token_next_by_instance(0, psql.Parenthesis)

    name = identifier.token_next_by_type(0, ptokens.Name)
    columns = find_tokens_by_instance(parenthesis.tokens, psql.Identifier, True)
    return name, columns


def insert_find_table_info(stmt):
    """
    Given an INSERT query, find and unquote the table name and column identifiers
    :param stmt: parsed statement tree from sqlparse
    :type stmt: sqlparse.sql.TokenList
    :return: the string of the table name, and the strings for the column names of the table
    """
    name, columns = insert_find_table_info_tokens(stmt)

    name = str(name).strip('"`')
    columns = [str(col_identifier.tokens[0]).strip('"`') for col_identifier in columns]

    return name, columns


def insert_replace_table_name(stmt, table_name):
    """
    Given an INSERT query, find and replace the table name.
    :param stmt: parsed statement tree from sqlparse
    :type stmt: sqlparse.sql.TokenList
    :param table_name: table name for the new INSERT query
    :type table_name: str
    """
    name, columns = insert_find_table_info_tokens(stmt)
    name.value = table_name


def insert_find_values(stmt):
    """
    Find the tokens representing the VALUES to insert in an INSERT query.
    :param stmt: parsed statement tree from sqlparse
    :type stmt: sqlparse.sql.TokenList
    :return: list of lists of string values to be inserted
    """
    values_keyword = stmt.token_next_match(0, ptokens.Keyword, 'VALUES')
    if values_keyword is None:
        return []

    parentheses = find_tokens_by_instance(stmt.tokens[stmt.token_index(values_keyword):],
        psql.Parenthesis)
    value_sets = []
    for parenthesis in parentheses:
        values = find_tokens_by_type(parenthesis.tokens, ptokens.Literal, recursive=True)
        values = [str(value).strip('"`\'') for value in values]
        value_sets.append(values)
    return value_sets


def update_find_data(stmt):
    """
    Extracts data relating to an UPDATE query.

    @returns table_name, col_value mapping dict, where_columns
    """
    table_name_token, comparisons, where_comparisons = update_find_tokens(stmt)

    table_name = str(table_name_token).strip('"`')
    col_values = OrderedDict()

    for comparison in comparisons:
        identifiers = find_tokens_by_instance(comparison.tokens, psql.Identifier) + \
                      find_tokens_by_type(comparison.tokens, ptokens.Literal)

        column = str(identifiers[0]).strip('"`')
        value = str(identifiers[1]).strip('"`\'')
        col_values[column] = value

    where_columns = find_columns_in_where_old(stmt)

    return table_name, col_values, where_columns


def update_find_tokens(stmt):
    """
    Finds table_name, comparisons, and where_comparisons objects within the statement given.
    """
    query_type_token = stmt.token_next_by_type(0, ptokens.DML)
    search_start_index = stmt.token_index(query_type_token) + 1

    identifier = stmt.token_next_by_instance(search_start_index, psql.Identifier)
    table_name_token = identifier.tokens[0]

    values_keyword = stmt.token_next_match(0, ptokens.Keyword, 'SET')
    search_start_index = stmt.token_index(values_keyword) + 1

    comparison = stmt.token_next_by_instance(search_start_index, psql.Comparison)
    identifier_list = stmt.token_next_by_instance(search_start_index, psql.IdentifierList)
    if identifier_list is None or (comparison is not None and
                                   stmt.token_index(comparison) < stmt.token_index(identifier_list)):
        comparisons = [comparison]
    else:
        comparisons = find_tokens_by_instance(identifier_list.tokens, psql.Comparison)

    where = stmt.token_next_by_instance(0, psql.Where)
    where_comparisons = find_tokens_by_instance(where.tokens, psql.Comparison) \
                        if where is not None else []

    return table_name_token, comparisons, where_comparisons


def update_replace_table_name(stmt, table_name):
    """
    Given an UPDATE query, find and replace the table name.
    :param stmt: parsed statement tree from sqlparse
    :type stmt: sqlparse.sql.TokenList
    :param table_name: table name for the new INSERT query
    :type table_name: str
    """
    table_name_token, comparisons, where_comparisons = update_find_tokens(stmt)
    table_name_token.value = table_name


def select_find_tables(stmt):
    """
    Find the table name of a SELECT query with only a single table
    :param stmt: parsed statement tree from sqlparse
    :type stmt: sqlparse.sql.TokenList
    :return: the table name as a string
    """
    from_token = stmt.token_next_by_instance(0, psqle.From)
    identifiers = find_tokens_by_instance(from_token.tokens, psql.Identifier)

    tables = set()
    for identifier in identifiers:
        # Note: we only accept single-database queries, so
        # SELECT * FROM yaleplus.Students, oci.Courses... does not work
        name = identifier.token_next_by_type(0, ptokens.Name)
        tables.add(str(name).strip('"`'))
    return tables


def select_find_columns(stmt, default_table):
    """
    Find the columns to be selected from a query
    :param stmt: parsed statement tree from sqlparse
    :type stmt: sqlparse.sql.TokenList
    :param default_table: the default table's name (for columns not in the form of table.column)
    :return: a list of [(table, column), ...] as strings. * is returned as (None, '*')
    """
    search_start_index = stmt.token_index(stmt.token_next_match(0, ptokens.DML, 'SELECT')) + 1

    # TODO(peterxu): This doesn't deal with SELECT (SELECT id...)
    next_token = stmt.token_next(search_start_index)

    # Could either be multiple identifiers, SELECT *, ..., or SELECT id FROM
    if isinstance(next_token, psql.IdentifierList):
        identifiers = find_tokens_by_instance(next_token.tokens, psql.Identifier) +\
                      find_tokens_by_type(next_token.tokens, ptokens.Wildcard)
    else: # isinstance(next_token, psql.Identifier) or next_token.ttype == ptokens.Wildcard:
        identifiers = [next_token]

    # Possible schemes
    # table.column AS alias: parsed as
    #   table (Name)
    #   column (Name)
    #   AS (Keyword)
    #   alias (Identifier)
    #       alias (Name)
    # In all cases, the last Name in the identifier is the column

    columns = set()
    for id in identifiers:
        if id.ttype == ptokens.Wildcard:
            columns.add((None, '*'))
            continue

        name_tokens = find_tokens_by_type(id.tokens, ptokens.Name)
        table = default_table if len(name_tokens) == 1 else name_tokens[0]
        column = name_tokens[-1]

        # For SELECT *; don't override SELECT table1.*
        if column == '*' and len(name_tokens) == 1:
            table = None

        columns.add((str(table), str(column),))

    return columns


def extract_type_data(type_str):
    # TODO(harryyu): This doesn't deal with ENUMs, CHARACTER SET, or NOT NULL
    type_data = {'length': 0}
    if ')' in type_str:
        type_data['type'] = type_str[:type_str.find(')') + 1]
        type_data['length'] = int(re.search('\(.*\)', type_data['type']).group().strip('()'))
        type_data['type'] = type_str[:type_str.find('(')]
    elif ' ' in type_str:
        type_data['type'] = type_str[:type_str.find(' ') + 1]
    else:
        type_data['type'] = type_str
    return type_data


def retrieve_schemas(conn):
    """
    Gathers schema data from the raw (non-wrapped connection).

    :type conn: pymysql.connections.Connection
    :returns two dicts, one containing a table_name -> {'col' -> col_info} and the other containing
             table_name -> num_rows
    :rtype tuple of dict, integer
    """
    cur = conn.cursor()
    cur.execute('SHOW TABLES')
    tables = [row[0] for row in cur.fetchall()]

    cur = conn.cursor(pymysql.cursors.DictCursor)

    tables_info = {}
    primary_keys = {}

    for table in tables:
        cur.execute('SHOW COLUMNS FROM {}'.format(table))
        tables_info[table] = OrderedDict()
        for col_info in cur.fetchall():
            column_name = col_info[u'Field']
            tables_info[table][column_name] = extract_type_data(col_info[u'Type'])
            if 'PRI' in col_info[u'Key']:
                primary_keys[table] = column_name

    cur.execute("SELECT TABLE_NAME, TABLE_ROWS FROM information_schema.tables WHERE TABLE_SCHEMA = "
                "'{}'".format(conn.db))
    tables_num_rows = {col_info[u'TABLE_NAME']: int(col_info['TABLE_ROWS'])
                       for col_info in cur.fetchall()}

    return tables_info, tables_num_rows, primary_keys


def group_schemas(tables_info, tables_num_rows):
    """
    Groups the schemas of the different tables by table name so that we have a dict of form:
    'table_name' => [(subtable_number, subtable_schema, num_rows)]

    For example, there may be tables test_table, test_table__1, test_table__2 which are part
    of the same sequence of tables.

    tables_info is a dict of form:
    'table_name' => {'column_name' => {column_info}, ...}
    """
    grouped_tables_info = defaultdict(list)

    for table_name, cols_info in tables_info.iteritems():
        if '__' in table_name:
            base_table_name = table_name[:table_name.find('__')]
            index = int(table_name[table_name.find('__') + 2:])
        else:
            base_table_name = table_name
            index = -1
        grouped_tables_info[base_table_name].append((index, cols_info, tables_num_rows[table_name]))

    grouped_tables_info = {table_name: sorted(sub_tables, reverse=True)
                           for (table_name, sub_tables) in grouped_tables_info.iteritems()}
    return grouped_tables_info


def find_minimum_types_for_values(value_sets):
    """
    Calculates the absolute minimum schema required for the values to be set.

    Returns the types (int/double/varchar) as well as the minimum lengths.
    :rtype list of string, list of int
    """

    # Initialize for number of columns
    num_columns = len(value_sets[0])
    col_types = ['tinyint'] * num_columns
    col_lengths = [0] * num_columns

    for values in value_sets:
        # We drop down the type to a more general type once it fails on anything
        for col_index, value in enumerate(values):
            if col_types[col_index].endswith('int'):
                try:
                    abs_int_value = abs(int(value))
                    if abs_int_value < 128:
                        col_types[col_index] = 'tinyint'
                    elif abs_int_value < 32768:
                        col_types[col_index] = 'smallint'
                    elif abs_int_value < 8388608:
                        col_types[col_index] = 'mediumint'
                    elif abs_int_value < 2147483648:
                        col_types[col_index] = 'int'
                    else:
                        col_types[col_index] = 'bigint'

                except ValueError:
                    col_types[col_index] = 'double'

            if col_types[col_index] == 'double':
                try:
                    float(value)
                except ValueError:
                    col_types[col_index] = 'varchar'

            if col_types[col_index] == 'varchar':
                col_lengths[col_index] = max(len(value), col_lengths[col_index])

    return col_types, col_lengths


def generate_column_definitions(col_info):
    """
    Based on the given columns, generates a list of strings representing lines to be added
    to ALTER TABLE or CREATE TABLE.

    :param col_info: OrderedDict with column_name keys and values of dicts containing
                     'type' and 'length'
    :type col_info: OrderedDict
    :return: a list of column definitions
    :rtype: list of str
    """
    column_definitions = []
    for col_name, col_type_info in col_info.iteritems():
        col_type = col_type_info['type']
        col_length = col_type_info['length']
        if col_type == 'varchar':
            col_length = max(col_length, 255)
            col_str = '{} {}({}) NULL'.format(col_name, col_type, col_length)
        else:
            col_str = '{} {} NULL'.format(col_name, col_type)
        column_definitions.append(col_str)

    return column_definitions


def generate_create_table(table_name, table_schema):
    col_strs = generate_column_definitions(table_schema)
    return 'CREATE TABLE {} (\n{}\n)'.format(table_name, ',\n'.join(col_strs))


def generate_alter_table(table_name, add_column_schema, modify_column_schema):
    """
    Generate an alter table query that adds columns or changes column sizes as
     needed.
    :param table_name: name of table to change
    :param add_column_schema: schema of the columns to add with ADD COLUMN in the
                              form of an OrderedDict with column_name keys and values
                              of dicts containing 'type' and 'length'
    :type add_column_schema: OrderedDict
    :param modify_column_schema: schema of the columns to add with MODIFY COLUMN in the
                                 form of an OrderedDict with column_name keys and values
                                 of dicts containing 'type' and 'length'
    :type modify_column_schema: OrderedDict
    :return:
    """
    add_strs = generate_column_definitions(add_column_schema)
    add_strs = ['ADD COLUMN ' + add_line for add_line in add_strs]
    modify_strs = generate_column_definitions(modify_column_schema)
    modify_strs = ['MODIFY COLUMN ' + line for line in modify_strs]
    lines = add_strs + modify_strs
    return 'ALTER TABLE {}\n{}'.format(table_name, ',\n'.join(lines))


def flexibility_score(column_info):
    """
    Converts the string of the type name to how flexible it is. When deciding whether to change
    the schema, higher flexibility always wins.
    """
    type_flexibility = {
        'tinyint': 100100,
        'smallint': 100200,
        'mediumint': 100300,
        'int': 100400,
        'bigint': 100500,
        'double': 200000,
        'varchar': 300000
    }
    return type_flexibility[column_info['type']] + column_info['length']


def generate_new_schema(existing_schema, query_schema):
    """
    Checks if the existing schema is sufficient for the query; if it is, return None.
    Otherwise, return a schema for a CREATE new table statement, as well as one for just
    ALTER statements.

    :type existing_schema: dict
    :type query_schema: dict
    :return: three dicts, one with a schema for CREATE, one with a schema for columns to
             be ADDed in an ALTER query, and finally one with a schema for columns to be
             MODIFYed in an ALTER query
    :rtype: (dict, dict, dict)
    """
    create_schema = OrderedDict()
    add_schema = OrderedDict()
    modify_schema = {}

    for column_name, existing_column in existing_schema.iteritems():
        if column_name not in query_schema:
            create_schema[column_name] = existing_column
            continue

        query_column = query_schema[column_name]
        if flexibility_score(query_column) > flexibility_score(existing_column):
            create_schema[column_name] = query_column
            modify_schema[column_name] = query_column
        else:
            create_schema[column_name] = existing_column

    for column_name, query_column in query_schema.iteritems():
        if column_name not in create_schema:
            if query_column['type'].endswith('int'):
                # Start off all int types as INT rather than TINYINT, etc.
                query_column['type'] = 'int'
            create_schema[column_name] = query_column
            add_schema[column_name] = query_column

    return create_schema, add_schema, modify_schema

def make_real_table_name(table_name, table_index):
    if table_index >= 0:
        return '{}__{}'.format(table_name, table_index)
    else:
        return table_name

def generate_triggers(old_table, new_table, shared_columns):
    """
    Creates a query to create triggers for new columns.
    """
    cols = ', '.join(shared_columns)
    cols_new = ', '.join(['NEW.' + col for col in shared_columns])

    insert_trigger = "CREATE TRIGGER {source_table}_insert AFTER INSERT ON {source_table} \n" \
                     "FOR EACH ROW BEGIN \n" \
                     "  IF (@disable_triggers IS NULL) THEN \n" \
                     "      INSERT INTO {dest_table} ({cols}) VALUES ({new_plus_cols}); \n" \
                     "  END IF; \n" \
                     "END;".format(source_table=new_table, dest_table=old_table,
                                   cols=cols, new_plus_cols=cols_new)
    return insert_trigger

def generate_select_arguments(table_name, table_schemas, columns_present=None):
    """
    Generates the parameters necessary for forward propagating a table into the delta tables.
    :param table_name: name of the (user-facing) table
    :param table_schemas: the schema of the current user-facing table, in the form of
                          schema = [(version, [('column name' => column info)...)
    :param columns_present: limit the columns examined to only these columns, in the form of
                            (table_name, column_name) tuples. Set to None for all columns
    :type columns_present: (str, str)
    :return: first_table_name (the left-most table that isn't joined on),
             a dictionary matching column names to a list of tables that have it: note that
             the list contains the oldest table versions first
             join_tables (the set of tables to left join on)
    :rtype: (str, defaultdict of list, set of str)
    """

    last_table_columns = {}
    # Stores the table columns that are covered by tables used so far
    column_versions = defaultdict(list)

    first_table_version = None
    for table_info in reversed(table_schemas):
        table_version = table_info[0]
        columns = table_info[1]
        first_table_version = table_version if first_table_version is None else first_table_version

        real_table_name = make_real_table_name(table_name, table_version)

        # Figure out which versions need to be included for queries involving each
        # column
        for column_name, type_info in columns.iteritems():
            # Only examine relevant columns
            if columns_present is None or (table_name, column_name,) in columns_present:
                # This column was created in this table-version
                if column_name not in last_table_columns:
                    column_versions[column_name].append(real_table_name)

                # This column was modified in this table-version
                elif type_info != last_table_columns[column_name]:
                    column_versions[column_name].append(real_table_name)

                # Otherwise, this column is the same as the previous version; all changes are back
                # propagated so we don't need to do anything

        last_table_columns = columns

    first_table = make_real_table_name(table_name, first_table_version)
    join_tables = set(itertools.chain.from_iterable(column_versions.itervalues()))
    join_tables.remove(first_table)

    return first_table, column_versions, join_tables


def make_join_string(first_table, join_tables, primary_key):
    """
    Make a join string for the bracketed parts in SELECT ... FROM [...] or UPDATE [...]
    :param first_table: earliest version of the table
    :param join_tables: other versions to join with
    :param primary_key: primary key to join on
    :return: a SQL query part similar to '`a` LEFT OUTER JOIN `b` USING `id` LEFT OUTER JOIN...'
    """
    left_outer_joins = ['LEFT OUTER JOIN `{table}` ON '
                        '`{orig_table}`.`{primary_key}` = `{table}`.`{primary_key}`'.format(
        table=join_table, orig_table=first_table, primary_key=primary_key)
                        for join_table in join_tables]
    return '`{table}` {joins}'.format(table=first_table, joins=' '.join(left_outer_joins))


def make_columns_string():
    """
    Make a columns string for the part in between SELECT and FROM. Can also be
    used for other fields.
    :return:
    """


def generate_propagate_sql(latest_version_table_name, table_name, table_schemas, primary_key,
                           where):
    """
    Generates an INSERT INTO... SELECT query for version migrations.
    :param latest_version_table_name: name of the latest version into which we're inserting
    :param table_name: user-facing table name
    :param table_schemas: the schema of the current table, in the form of
                          schema = (version, [('column name' => column info)...)
    :param primary_key: column of the primary key of the table
    :type primary_key: str
    :param where: the parsed group containing the WHERE conditions
    :type where: sqlparse.sql.Where
    :return: string of INSERT INTO ... SELECT sql
    :rtype: str
    """
    first_table, copy_columns, join_tables = generate_select_arguments(
        table_name, table_schemas)

    selects = []
    for column, tables in copy_columns.iteritems():
        reversed_tables = reversed(tables)  # We want the latest versions to come first for COALESCE
        columns_str = ', '.join(['{}.{}'.format(table, column) for table in reversed_tables])
        if len(tables) == 1:
            selects.append('{} AS {}'.format(columns_str, column))
        else:
            selects.append('COALESCE({}) AS {}'.format(columns_str, column))

    selects = ', '.join(selects)

    left_outer_joins = make_join_string(first_table, join_tables, primary_key)

    new_where = copy.deepcopy(where)
    replace_where_identifiers(new_where, {table_name: copy_columns})

    full_select = 'SELECT {selects} FROM {table} {left_outer_joins} {where_str}'.format(
        selects=selects, table=table_name, left_outer_joins=left_outer_joins, where_str=new_where)

    insert_columns = ', '.join(copy_columns.keys())
    insert_query = 'INSERT IGNORE INTO {insert_table_name} ({insert_columns}) {select}'.format(
        insert_table_name=latest_version_table_name, insert_columns=insert_columns,
        select=full_select)

    return insert_query


def infer_table(column, tables_columns):
    """
    Infers what table a column is from.

    :type column: str
    :param tables_columns: a dict of 'table' => {'col' => ['subtable1', ...]}
    :type tables_columns: dict of (str -> dict of (str -> list of str))
    :return: the string name of the table; None if it doesn't exist, or False on conflict
    :rtype: str | None | False
    """
    table_found = None

    for table, columns in tables_columns.iteritems():
        if column in columns:
            # We've already found a table! This reference is ambiguous, so we just leave it
            if table_found is not None:
                return False
            else:
                table_found = table

    return table_found

def convert_comparison_for_multi_table(comparison_token, parent_token, column, tables):
    comparison_token_index = parent_token.token_index(comparison_token)
    parenthesis = psql.Parenthesis()
    parent_token.tokens[comparison_token_index] = parenthesis

    parenthesis.tokens.append(psql.Token(ptokens.Punctuation, '('))

    tables_to_check_for_null = []

    for table in reversed(tables):
        sub_parenthesis = psql.Parenthesis()
        sub_parenthesis.tokens.append(psql.Token(ptokens.Punctuation, '('))

        new_comparison_token = copy.copy(comparison_token)
        identifier = new_comparison_token.token_next_by_instance(0, psql.Identifier)
        identifier_index = new_comparison_token.token_index(identifier)
        new_comparison_token.tokens[identifier_index] = psql.Identifier([
            psql.Token(ptokens.Name, table),
            psql.Token(ptokens.Punctuation, '.'),
            psql.Token(ptokens.Name, column)
        ])
        identifier = new_comparison_token.tokens[identifier_index]
        sub_parenthesis.tokens.append(new_comparison_token)

        for null_column_table in tables_to_check_for_null:
            identifier = psql.Identifier([
                psql.Token(ptokens.Name, null_column_table),
                psql.Token(ptokens.Punctuation, '.'),
                psql.Token(ptokens.Name, column)
            ])

            null_comparison = psql.Comparison([
                identifier,
                psql.Token(ptokens.Whitespace, ' '),
                psql.Token(ptokens.Keyword, 'IS'),
                psql.Token(ptokens.Whitespace, ' '),
                psql.Token(ptokens.Keyword, 'NULL')
            ])

            sub_parenthesis.tokens.extend([
                psql.Token(ptokens.Whitespace, ' '),
                psql.Token(ptokens.Keyword, 'AND'),
                psql.Token(ptokens.Whitespace, ' '),
                null_comparison
            ])

        sub_parenthesis.tokens.append(psql.Token(ptokens.Punctuation, ')'))

        tables_to_check_for_null.append(table)

        if len(parenthesis.tokens) > 1:  # We have more than the starting parenthesis
            parenthesis.tokens.extend([
                psql.Token(ptokens.Whitespace, ' '),
                psql.Token(ptokens.Keyword, 'OR'),
                psql.Token(ptokens.Whitespace, ' '),
            ])
        parenthesis.tokens.append(sub_parenthesis)

    parenthesis.tokens.append(psql.Token(ptokens.Punctuation, ')'))


def replace_identifiers(tokens, column_versions):
    """
    Replaces the identifiers in the tokens list. Note that while this only works for WHERE
    clauses right now, it's very easily adaptable. The only WHERE-specific code is right after
    the "else"

    :param where_token: the Where group token group
    :type where_token: sqlparse.sql.Where
    :param column_versions: a dict of 'table' => {'col' => ['subtable1', ...]}
    :type column_versions: dict of (str -> dict of (str -> list of str))
    :rtype:
    """
    identifiers = find_identifiers_with_name_sub_token(tokens)
    for identifier in identifiers:
        table, column = separate_table_and_column(identifier, None)
        if table is None:
            table = infer_table(column, column_versions)
        if table is None or not table:
            continue

        real_table_candidates = column_versions[table][column]

        sql_fragments = ['`{table}`.`{column}`'.format(
            table=real_table, column = column) for real_table in real_table_candidates]

        if len(sql_fragments) == 1:
            identifier.tokens = psqle.parse(sql_fragments[0])
        else:
            identifier.tokens = psqle.parse('COALESCE({}) AS {}'.format(', '.join(sql_fragments),
                column))


def replace_where_identifiers(where_token, column_versions):
    """
    Replaces the identifiers in the tokens list. Note that while this only works for WHERE
    clauses right now, it's very easily adaptable. The only WHERE-specific code is right after
    the "else"

    :param where_token: the Where group token group
    :type where_token: sqlparse.sql.Where
    :param column_versions: a dict of 'table' => {'col' => ['subtable1', ...]}
    :type column_versions: dict of (str -> dict of (str -> list of str))
    :rtype:
    """

    identifiers = find_identifiers_with_name_sub_token(where_token.tokens)

    for identifier in identifiers:
        table, column = separate_table_and_column(identifier, None)
        if table is None:
            table = infer_table(column, column_versions)
        if table is None or not table:
            continue

        real_table_candidates = column_versions[table][column]
        if len(real_table_candidates) == 1:
            real_table = real_table_candidates[0]

            identifier.tokens = [
                psql.Token(ptokens.Name, real_table),
                psql.Token(ptokens.Punctuation, '.'),
                psql.Token(ptokens.Name, column)
            ]
        else:
            # TODO: note this doesn't deal with IS NULL or IS NOT NULL comparators
            output = find_comparison_with_identifier(where_token, identifier)
            # If output is null, we have a "column = 123": encapsulate identifier, operator, next token
            if output is None:
                comparison = psql.Comparison()
                operator = where_token.token_next(where_token.token_index(identifier))
                value = where_token.token_next(where_token.token_index(operator))
                comparison.tokens = [
                    identifier,
                    psql.Token(ptokens.Whitespace, ' '),
                    operator,
                    psql.Token(ptokens.Whitespace, ' '),
                    value
                ]
                del where_token.tokens[where_token.token_index(value)]
                del where_token.tokens[where_token.token_index(operator)]
                id_index = where_token.token_index(identifier)
                del where_token.tokens[id_index]
                where_token.tokens.insert(id_index, comparison)

                # Fetch it again
                output = find_comparison_with_identifier(where_token, identifier)

            comparison_token, parent_token = output
            convert_comparison_for_multi_table(comparison_token, parent_token, column,
                real_table_candidates)