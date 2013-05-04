__author__ = 'User'

import sqlparse
import sqlparse.sql
import threading

from flexidata import *

conn = Connection(original_conn)

def get_table_migration_states(conn):
    """
    Gets the table migration status from the database.

    :param conn:
    :type conn: flexidata.Connection
    :return: a dict of real_table_name => last_id_processed
    :rtype: dict of (str, int)
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
    Migrates the next blah IDs to the subtable. This doesn't check when it's done though!
    Be careful!

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

    new_last_processed = cur.lastrowid
    # TODO(harryyu) Note that if there are gaps in the primary key, rows can be processed twice
    cur.execute("INSERT INTO migrations (destination, last_id_processed) VALUES ('{1}', {2}) \n"
                "ON DUPLICATE KEY UPDATE SET last_id_processed = {2}".format(subtable_name,
                                                                             new_last_processed))
    conn.commit()
    return new_last_processed

class MigrateThread(threading.Thread):

    def __init__(self):
        super(MigrateThread, self).__init__()
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()

    def run(self, seconds_per_check, rows_per_check, busy_processes_threshold, table_priority):
        """
        Runs the thread to automatically run DB migrations.

        :param seconds_per_check: How many seconds (including decimals) to wait between checks
        :param rows_per_check: How many rows to update per successful check
        :param busy_processes_threshold: How many processes running to disallow the run
        :param table_priority: List of tables to process in order
        :type table_priority: list | None
        """
        global conn
        tables_last_processed = get_table_migration_states(conn)

