__author__ = 'User'

'''
High level outline:
- Scan the tables' schemas
- Store the number of rows in each table for the last update
- At 100 rows, and every doubling thereafter, we scan
'''
import pymysql

from flexidata import *

conn = Connection(original_conn)

# Get a raw MySQL DictCursor
cur = conn.conn.cursor(pymysql.cursors.DictCursor)
cursor = conn.cursor()

for table, versions in conn.schemas.iteritems():
    # Only try to compact when there's already only one version
    # because that's when the PROCEDURE ANALYSE will work
    #if len(versions) != 1:
    #    continue

    version = versions[0]
    version_num, cur_columns, num_rows = version
    version_table = make_real_table_name(table, version_num)

    analyse = 'SELECT * FROM `{table}` PROCEDURE ANALYSE()'.format(
        table=version_table)
    cur.execute(analyse)
    columns = cur.fetchall()

    new_columns = cur_columns

    # Annotate each of the fields with their current types and new types
    for column_info in columns:
        column = column_info['Field_name'].split('.')[-1]
        cur_length, cur_type = cur_columns[column].itervalues()

        #print column_info
        #print cur_length, cur_type

        if cur_type.endswith('int'):
            max_abs_value = max(abs(int(column_info['Max_value'])), abs(int(column_info['Min_value'])))
            numeric_types = [
                (128, 'tinyint', 4),
                (32768, 'smallint', 6),
                (8388608, 'mediumint', 8),
                (2147483648, 'int', 11),
                (9223372036854775808, 'bigint', 20)
            ]

            # Determine the right type, and shrink the column if needed with a 2x allowance
            for max_type_value, type_name, type_length in numeric_types:
                if max_type_value >= 2 * max_abs_value:
                    new_type = type_name
                    new_length = type_length
                    break

            if new_type != cur_type:
                new_columns[column] = {'length': type_length, 'type': new_type}
                # The length doesn't really matter for numeric types

        elif cur_type.endswith('char'):
            max_length = column_info['Max_length']

            # Shrink to 1.5 times max_length if current size >= 2 * max_length
            if cur_length >= 2 * max_length:
                # If more than 1.5 times, multiply by 2
                new_columns[column] = {'length': int(round(max_length * 1.5)), 'type': cur_type}

    # Do the update if needed
    cursor._prepare_schema(table, new_columns)

