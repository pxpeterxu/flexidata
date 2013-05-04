__author__ = 'User'

import sqlparse
import sqlparse.sql
import threading
import time

from flexidata import *

conn = Connection(original_conn)

def get_table_migration_states(conn):
    """
    Gets the table migration status from the database.

    :param conn:
    :type conn: flexidata.Connection
    :return: a dict of real_table_name => last_id_processed
    :rtype: dict of (str, tuple)
    """
    cur = conn.cursor()
    if 'migrations' not in conn.schemas:
        create_migrations_sql = '''
            CREATE TABLE `migrations` (
                `id` INT(10) UNSIGNED NOT NULL AUTO_INCREMENT,
                `base_table` VARCHAR(255) NULL DEFAULT NULL,
                `subtable_index` SMALLINT(5) UNSIGNED NULL DEFAULT NULL,
                `last_id_processed` INT(10) UNSIGNED NULL DEFAULT NULL,
                PRIMARY KEY (`id`),
                UNIQUE INDEX `base_table` (`base_table`)
            )
            ENGINE=InnoDB;
            '''
        cur.execute(create_migrations_sql)
        conn.commit()

    conn.refresh_schemas()
    cur.execute('SELECT base_table, subtable_index, last_id_processed FROM migrations')
    last_processed = {row[0]: (row[1], row[2]) for row in cur.fetchall()}

    for table_name, subtables in conn.schemas.iteritems():
        if len(subtables) > 1:
            subtable_index = subtables[0][0]
            if table_name not in last_processed or last_processed[table_name][0] != subtable_index:
                last_processed[table_name] = (subtable_index, 0)

    return last_processed

def migrate_table(conn, base_table, subtable_index, last_processed, num_to_process):
    """
    Migrates the next num_to_process IDs to the subtable.

    :type conn: flexidata.Connection
    """
    subtable_name = make_real_table_name(base_table, subtable_index)
    primary_key = conn.primary_keys[base_table]
    where_clause = sqlparse.parse('SELECT * FROM a_table WHERE {} > {}'.format(primary_key,
                                                                               last_processed))[0]
    where_clause = where_clause.token_next_by_instance(0, sqlparse.sql.Where)
    propagate_sql = generate_propagate_sql(subtable_name, base_table, conn.schemas[base_table],
                                           primary_key, where_clause)
    propagate_sql += ' LIMIT 0, {}'.format(num_to_process)

    cur = conn.conn.cursor()  # We use the raw connection
    cur.execute('SET @disable_triggers = 1;')
    cur.execute(propagate_sql)
    num_processed = cur.rowcount
    cur.execute('SET @disable_triggers = NULL;')

    conn.commit()
    new_last_processed = num_processed + last_processed

    if num_processed == 0:
        cur.execute('SELECT {0} FROM {1} ORDER BY {0} DESC LIMIT 0, 1'.format(primary_key,
                                                                              base_table))
        base_last_row = cur.fetchone()[0]
        cur.execute('SELECT {0} FROM {1} ORDER BY {0} DESC LIMIT 0, 1'.format(primary_key,
                                                                              subtable_name))
        subtable_last_row = cur.fetchone()[0]
        done_with_table = base_last_row == subtable_last_row
    else:
        done_with_table = False

    if done_with_table:
        cur.execute("DELETE FROM migrations WHERE base_table = '{}'".format(base_table))
        for index in range(0, subtable_index + 1):
            name = make_real_table_name(base_table, index)
            if index > -1:
                cur.execute('DROP TRIGGER {}_insert'.format(name))
                cur.execute('DROP TRIGGER {}_update'.format(name))

        for index in range(-1, subtable_index):
            name = make_real_table_name(base_table, index)
            cur.execute('DROP TABLE {}'.format(base_table))
        cur.execute('RENAME TABLE {} TO {}'.format(subtable_name, base_table))
    else:
        # TODO(harryyu) Note that if there are gaps in the primary key, rows can be processed twice
        save_state_sql = ("INSERT INTO migrations (base_table, subtable_index, last_id_processed) \n"
                          "VALUES ('{0}', {1}, {2}) \n"
                          "ON DUPLICATE KEY UPDATE subtable_index = {1}, last_id_processed = {2}"
                          .format(base_table, subtable_index, new_last_processed))
        cur.execute(save_state_sql)
    conn.commit()
    return new_last_processed

class MigrateThread(threading.Thread):

    def __init__(self, seconds_per_check, rows_per_check, busy_processes_threshold=None):
        """

        :param seconds_per_check: How many seconds (including decimals) to wait between checks
        :param rows_per_check: How many rows to update per successful check
        :param busy_processes_threshold: How many processes running to disallow the run
        """
        super(MigrateThread, self).__init__()
        self._stop = threading.Event()
        self.seconds_per_check = seconds_per_check
        self.rows_per_check = rows_per_check
        self.busy_processes_threshold = busy_processes_threshold

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()

    def run(self):
        """
        Runs the thread to automatically run DB migrations.
        """
        global conn
        cur_table = None
        while not self.stopped():
            states = get_table_migration_states(conn)
            if cur_table is None or cur_table not in states:
                # noinspection PyTypeChecker
                if len(states) > 0:
                    cur_table = next(states.iterkeys())  # First table, whatever

            if cur_table is not None:
                last_processed_id = states[cur_table][1]
                subtable_index = states[cur_table][0]
                migrate_table(conn, cur_table, subtable_index, last_processed_id, self.rows_per_check)

            time.sleep(self.seconds_per_check)

thread = MigrateThread(seconds_per_check=10, rows_per_check=20)
thread.start()