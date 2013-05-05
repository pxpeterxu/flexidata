import sys
import flexidata
import pymysql
import settings
import time

if len(sys.argv) < 3:
    print '''
Usage: python test_flexidata.py sql-file flexidata|pymysql
This script
1. Reads a SQL file into memory
2. Executes the statements one-by-one
3. Times how long each statement took
'''
    exit()

filename = sys.argv[1]
conn_type = sys.argv[2]

f = open(filename)
sql_statements = f.readlines()
f.close()

original_conn = pymysql.connect(
    db=settings.flexidata_database,
    user=settings.flexidata_username,
    passwd=settings.flexidata_password,
    host=settings.flexidata_host)

# Disable query cache for
original_cursor = original_conn.cursor()
original_cursor.execute('SET SESSION query_cache_type = OFF')
original_conn.commit()

if conn_type == 'mysql':
    conn = original_conn
elif conn_type == 'flexidata':
    conn = flexidata.Connection(original_conn)
    conn.hint_primary_key('Students2', 'id')
cursor = conn.cursor()

times = []
start_time = time.time()
for sql_statement in sql_statements:
    cursor.execute(sql_statement)

    if sql_statement.startswith('SELECT'):
        cursor.fetchall()
    else:
        conn.commit()

    times.append(time.time() - start_time)

i = 1
for time in times:
    print "{i},{time}".format(i=i, time=time)
    i += 1


