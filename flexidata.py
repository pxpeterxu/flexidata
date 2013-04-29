__author__ = 'User'

import sqlparse
from sqlparse import sql as psql
from sqlparse import tokens as ptokens

import pymysql
import re

from collections import defaultdict, OrderedDict
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

    def _refresh_schemas(self):
        self.schemas, self.num_rows = retrieve_schemas(self.conn)
        self.schemas = group_schemas(self.schemas, self.num_rows)
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
                                                       shared_columns, [])
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
        table_name, row_data, where_columns = update_find_data(stmt)
        types, lengths = find_minimum_types_for_values([row_data.values()])

        query_minimum_schema = OrderedDict()
        for info in zip(row_data.keys(), types, lengths):
            query_minimum_schema[info[0]] = {'type': info[1], 'length': info[2]}

        return self._prepare_schema(table_name, query_minimum_schema)

    def _prepare_for_select(self, stmt):
        # Get basic information
        table_name = select_find_table_name(stmt)
        columns = select_find_columns(stmt, table_name)

        # Identify the latest version of each table listed that we need

        # Start
        schema = self.conn.schemas[table_name]




        #return self._prepare_schema(table_name, query_minimum_schema)

    def execute(self, query, args=None):
        stmts = sqlparse.parse(query)

        for stmt in stmts:
            if stmt.token_next_match(0, ptokens.DML, 'INSERT') is not None:
                real_table_name = self._prepare_for_insert(stmt)
                insert_replace_table_name(stmt, real_table_name)
            if stmt.token_next_match(0, ptokens.DML, 'UPDATE') is not None:
                real_table_name = self._prepare_for_update(stmt)
                update_replace_table_name(stmt, real_table_name)

        stmts = [str(stmt) for stmt in stmts]
        query = ';\n'.join(stmts)
        return self.cursor.execute(query, args)

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


def find_columns_in_where(stmt):
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


def find_columns_after_keyword(stmt):
    """
    Find all column names in a statement following a given keyword
    :param stmt: parsed statement from sqlparse
    :return: list of columns after the keyword by their string names
    """
    yield


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

    where_columns = find_columns_in_where(stmt)

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


def select_find_table_name(stmt):
    """
    Find the table name of a SELECT query with only a single table
    :param stmt: parsed statement tree from sqlparse
    :type stmt: sqlparse.sql.TokenList
    :return: the table name as a string
    """
    search_start_index = stmt.token_index(stmt.token_next_match(0, ptokens.Keyword, "FROM")) + 1
    identifier = stmt.token_next_by_instance(search_start_index, psql.Identifier)
    name = identifier.token_next_by_type(0, ptokens.Name)
    return str(name).strip('"`')


def select_find_columns(stmt, default_table):
    """
    Find the columns to be selected from a query
    :param stmt: parsed statement tree from sqlparse
    :type stmt: sqlparse.sql.TokenList
    :param default_table: the default table's name (for columns not in the form of table.column)
    :return: a list of [(table, column), ...] as strings. * is returned as (None, '*')
    """
    search_start_index = stmt.token_index(stmt.token_next_match(0, ptokens.Keyword, "SELECT")) + 1

    # TODO(peterxu): This doesn't deal with SELECT (SELECT id...)
    identifierlist = stmt.token_next_by_instance(search_start_index, psql.IdentifierList)
    identifier = stmt.token_next_by_instance(search_start_index, psql.Identifier)

    # If there's no identifierlist, or if the identifier comes before the identifierllist,
    # we have a SELECT single_column FROM table
    if identifierlist is None or (identifier is not None
                            and stmt.token_index(identifier) < stmt.token_index(identifierlist)):
        identifiers = [identifier]
    else:
        identifiers = find_tokens_by_instance(identifierlist, psql.Identifier)

    # Possible schemes
    # table.column AS alias: parsed as
    #   table (Name)
    #   column (Name)
    #   AS (Keyword)
    #   alias (Identifier)
    #       alias (Name)
    # In all cases, the last Name in the identifier is the column

    columns = []
    for id in identifiers:
        name_tokens = find_tokens_by_type(id, ptokens.Name)
        table = default_table if len(name_tokens) == 1 else name_tokens[0]
        column = name_tokens[-1]

        # For SELECT *; don't override SELECT table1.*
        if column == '*' and len(name_tokens) == 1:
            table = None

        columns.append((str(table), str(column),))

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
    for table in tables:
        cur.execute('SHOW COLUMNS FROM {}'.format(table))
        tables_info[table] = OrderedDict()
        for col_info in cur.fetchall():
            tables_info[table][col_info[u'Field']] = extract_type_data(col_info[u'Type'])

    cur.execute("SELECT TABLE_NAME, TABLE_ROWS FROM information_schema.tables WHERE TABLE_SCHEMA = "
                "'{}'".format(conn.db))
    tables_num_rows = {col_info[u'TABLE_NAME']: int(col_info['TABLE_ROWS'])
                       for col_info in cur.fetchall()}

    return tables_info, tables_num_rows


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
    :type value_sets
    :rtype list of string, list of int
    """

    # Initialize for number of columns
    num_columns = len(value_sets[0])
    col_types = ['int'] * num_columns
    col_lengths = [0] * num_columns

    for values in value_sets:
        # We drop down the type to a more general type once it fails on anything
        for col_index, value in enumerate(values):
            if col_types[col_index] == 'int':
                try:
                    int(value)
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
        'int': 100000,
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
            create_schema[column_name] = query_column
            add_schema[column_name] = query_column

    return create_schema, add_schema, modify_schema

def make_real_table_name(table_name, table_index):
    if table_index >= 0:
        return '{}__{}'.format(table_name, table_index)
    else:
        return table_name

def generate_triggers(old_table, new_table, shared_columns, unique_columns):
    """
    Creates a query to create triggers for new columns.
    """
    cols = ', '.join(shared_columns)
    cols_new = ', '.join(['NEW.' + col for col in shared_columns])

    unique_cols = ', '.join(unique_columns)
    unique_cols_old = ['OLD.' + col for col in unique_cols]

    insert_trigger = "CREATE TRIGGER {source_table}_insert AFTER INSERT ON {source_table} \n" \
                     "FOR EACH ROW INSERT INTO {dest_table} ({cols}) VALUES \n" \
                     "({new_plus_cols})".format(source_table=new_table, dest_table=old_table,
                                                cols=cols, new_plus_cols=cols_new)
    # update_trigger = "CREATE TRIGGER {source_table}_insert AFTER UPDATE ON {source_table} \n" \
    #                  "FOR EACH ROW BEGIN \n" \
    #                  "  DELETE FROM {dest_table} WHERE ({unique_cols}) = ({unique_cols_old});\n" \
    #                  "  REPLACE INTO {dest_table} ({cols}) VALUES ({new_plus_cols});" \
    #                  "END".format(source_table=new_table, dest_table=old_table, cols=cols,
    #                               new_plus_cols=cols_new, unique_cols=unique_cols,
    #                               unique_cols_old=unique_cols_old)
    update_trigger = ""
    return ';\n'.join((insert_trigger, update_trigger))

conn = Connection(original_conn)
cur = conn.cursor()

cur.execute("INSERT INTO test_table (sid, name, college, cash) VALUES"
            "(10210101, 'George Bush', 'Davenport', 9999999.54)")
conn.commit()
cur.execute("UPDATE test_table SET class_year = 1964 WHERE name = 'George Bush'")
conn.commit()
# cur.execute("INSERT INTO test_table (id, name, college, cash, class_year) VALUES "
#             "(909876542, 'Peter Xu', 'Morse', 'no mo money yo', '2014')")
# conn.commit()

