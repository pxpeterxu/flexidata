import pymysql
import settings
import sys

_, db, table = sys.argv

# Export a table for testing using test_flexidata.py
conn = pymysql.connect(
    db=db,
    user=settings.flexidata_username,
    passwd=settings.flexidata_password,
    host=settings.flexidata_host)


cursor = conn.cursor(pymysql.cursors.DictCursor)
cursor.execute('SELECT * FROM {table}'.format(table=table))
results = cursor.fetchall()

for result in results:
    query = 'INSERT INTO {table} ({columns}) VALUES ({values})'.format(table=table,
        columns=', '.join(result.keys()), values=', '.join(result.values()))
    print query


