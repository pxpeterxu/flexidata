__author__ = 'User'

import sqlparse
import sqlparse.sql

from flexidata import *

conn = Connection(original_conn)
cur = conn.cursor()

def get_table_migration_states(conn):
    """

    :param conn:
    :type conn: flexidata.Connection
    :return:
    :rtype:
    """
    cur = conn.cursor()
    if 'migrations' not in conn.schemas:
        create_migrations_sql = '''
            CREATE TABLE `migrations` (
                `id` INT(10) UNSIGNED NULL AUTO_INCREMENT,
                `destination` VARCHAR(255) NULL,
                `last_id_processed` INT UNSIGNED NULL,
                PRIMARY KEY (`id`),
                UNIQUE INDEX `destination` (`destination`)
            )'''
        cur.execute(create_migrations_sql)
        conn.commit()

    cur.execute('SELECT * FROM migrations')
    last_processed = cur.fetchall()
    for table_name, subtables in conn.schemas.itervalues():
        if len(subtables) > 0:
            subtable_name = make_real_table_name(table_name, subtables[0][0])
            last_processed[subtable_name] = 0

    return last_processed

def migrate_table(conn, subtable_name, last_processed, num_to_process):
    """
    :type conn: flexidata.Connection
    """
    base_table = get_base_table(subtable_name)
    primary_key = conn.primary_keys[base_table]
    where_clause = sqlparse.parse('SELECT * FROM a_table WHERE {} > {}'.format(primary_key,
                                                                               last_processed))
    where_clause = where_clause.token_next_by_instance(0, sqlparse.sql.Where)
    propagate_sql = generate_propagate_sql(subtable_name, base_table, conn.schemas[base_table],
                                           primary_key, where_clause)

    cur = conn.conn.cursor()  # We use the raw connection
    cur.execute('SET @disable_triggers = 1;')
    cur.execute(propagate_sql + ' LIMIT 0, {}'.format(num_to_process))
    cur.execute('SET @disable_triggers = NULL;')

    conn.commit()

    new_last_processed = last_processed + num_to_process
    # TODO(harryyu) Note that if there are gaps in the primary key, rows can be processed twice
    cur.execute("INSERT INTO migrations (destination, last_id_processed) VALUES ('{1}', {2}) \n"
                "ON DUPLICATE KEY UPDATE SET last_id_processed = {2}".format(subtable_name,
                                                                             new_last_processed))
    conn.commit()
    return new_last_processed