__author__ = 'User'

import sqlparse
from sqlparse import sql as psql
from sqlparse import tokens as ptokens

import pymysql
import re
import pyparsing

from collections import defaultdict
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

    def close(self):
        self.conn.close()

    def commit(self):
        self.conn.commit()

    def rollback(self):
        self.conn.rollback()

    def cursor(self):
        return Cursor(self.conn.cursor(), self)

    def _refresh_schemas(self):
        self.schemas = group_schemas(retrieve_schemas(self.conn))


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

    def _prepare_for_insert(self, stmt):
        self.conn._refresh_schemas()
        schemas = self.conn.schemas

        table_name, columns = insert_find_table_info(stmt)
        value_sets = insert_find_values(stmt)
        types, lengths = create_schema_from_values(value_sets)

        if table_name not in schemas:
            create_table = generate_create_table(table_name, columns, types, lengths)
            self.cursor.execute(create_table)

        else:
            table_name = make_insert_table_name(table_name, schemas[table_name])
            table_info = schemas[table_name][0][1]
            cols_to_create = []
            cols_to_modify = []

            # Table already exists; verify we can insert into it
            for i in range(0, len(types)):
                col_name = columns[i]
                col_type = types[i]
                col_length = lengths[i]

                if col_name not in table_info:
                    cols_to_create.append((col_name, col_type, col_length))
                    continue

                existing_type = table_info[col_name]['type']
                existing_length = table_info[col_name]['length']
                if (calculate_flexibility(col_type) > calculate_flexibility(existing_type) or
                    col_length > existing_length):
                    cols_to_modify.append((col_name, col_type, col_length))

            if cols_to_create or cols_to_modify:
                alter_table = generate_alter_table(table_name, cols_to_create, cols_to_modify)
                self.cursor.execute(alter_table)

        self.conn._refresh_schemas()

    def execute(self, query, args=None):
        stmts = sqlparse.parse(query)

        for stmt in stmts:
            if stmt.token_next_match(0, ptokens.DML, 'INSERT') is not None:
                self._prepare_for_insert(stmt)

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
    name, columns = insert_find_table_info_tokens(stmt)

    name = str(name).strip('"`')
    columns = [str(col_identifier.tokens[0]).strip('"`') for col_identifier in columns]

    return name, columns


def insert_replace_table_name(stmt, table_name):
    name, columns = insert_find_table_info_tokens(stmt)
    name.value = table_name


def insert_find_values(stmt):
    """
    :type stmt sqlparse.sql.Statement
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


def select_find_table_name(stmt):
    """
    :type stmt: sqlparse.sql.Statement
    """
    search_start_index = stmt.token_index(stmt.token_next_match(0, ptokens.Keyword, "FROM")) + 1
    identifier = stmt.token_next_by_instance(search_start_index, psql.Identifier)
    name = identifier.token_next_by_type(0, ptokens.Name)
    return str(name).strip('"`')


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
    """
    cur = conn.cursor()
    cur.execute('SHOW TABLES')
    tables = [row[0] for row in cur.fetchall()]

    cur = conn.cursor(pymysql.cursors.DictCursor)

    tables_info = {}
    for table in tables:
        cur.execute('SHOW COLUMNS FROM {}'.format(table))
        tables_info[table] = {col_info[u'Field']: extract_type_data(col_info[u'Type'])
                              for col_info in cur.fetchall()}

    return tables_info


def group_schemas(tables_info):
    """
    Groups the schemas of the different tables by table name so that we have a dict of form:
    'table_name' => [(subtable_number, subtable_schema)]

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
        grouped_tables_info[base_table_name].append((index, cols_info))

    grouped_tables_info = {table_name: sorted(sub_tables)
                           for (table_name, sub_tables) in grouped_tables_info.iteritems()}
    return grouped_tables_info


def create_schema_from_values(value_sets):
    """
    Calculates the absolute minimum schema required for the values to be set.

    Returns the types (int/double/varchar) as well as the minimum lengths.
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

    :param col_info: tuple with 3 elements for name, type, and length
    :type col_info: tuple
    :return: a list of column definitions
    :rtype: list of str
    """
    NAME = 0
    TYPE = 1
    LENGTH = 2

    column_definitions = []
    for col_info in col_info:
        if col_info[TYPE] == 'varchar':
            length = max(col_info[LENGTH], 255)
            col_str = '{} {}({}) NULL'.format(col_info[NAME], col_info[TYPE],
                      length)
        else:
            col_str = '{} {} NULL'.format(col_info[NAME], col_info[TYPE])
        column_definitions.append(col_str)

    return column_definitions


def generate_create_table(table_name, column_names, types, lengths):
    col_strs = generate_column_definitions(zip(column_names, types, lengths))
    return 'CREATE TABLE {} (\n{}\n)'.format(table_name, ',\n'.join(col_strs))


def generate_alter_table(table_name, columns_to_add, columns_to_modify):
    add_strs = generate_column_definitions(columns_to_add)
    add_strs = ['ADD COLUMN ' + add_line for add_line in add_strs]
    modify_strs = generate_column_definitions(columns_to_modify)
    modify_strs = ['MODIFY COLUMN ' + line for line in modify_strs]
    lines = add_strs + modify_strs
    return 'ALTER TABLE {}\n{}'.format(table_name, ',\n'.join(lines))


def calculate_flexibility(sql_type_name):
    """
    Converts the string of the type name to how flexible it is. When deciding whether to change
    the schema, higher flexibility always wins.
    """
    flexibility = {
        'int': 1,
        'double': 2,
        'varchar': 3
    }
    return flexibility[sql_type_name]


def make_insert_table_name(table_name, table_group):
    table_index = table_group[0][0]
    if table_index >= 0:
        return table_name + '__' + table_index
    else:
        return table_name


conn = Connection(original_conn)
cur = conn.cursor()

stmt = sqlparse.parse("INSERT INTO test_table (sid, name, college, cash) VALUES"
                      "(10210101, 'George Bush', 'Davenport', 9999999.54)")[0]
print_token_children(stmt)
insert_replace_table_name(stmt, 'blah_table')
print_token_children(stmt)

# cur.execute("INSERT INTO test_table (sid, name, college, cash) VALUES"
#             "(10210101, 'George Bush', 'Davenport', 9999999.54)")
# conn.commit()
# cur.execute("INSERT INTO test_table (id, name, college, cash, class_year) VALUES "
#             "(909876543,'Harry Yu', 'Saybrook', 12.34, '2014')")
# conn.commit()
# cur.execute("INSERT INTO test_table (id, name, college, cash, class_year) VALUES "
#             "(909876542, 'Peter Xu', 'Morse', 'no mo money yo', '2014')")
# conn.commit()

