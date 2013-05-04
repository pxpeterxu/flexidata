import pymysql
import settings
import sys

'''
4 main use cases:
1. Select a single person by netId
2. Search for the name of a person that starts with a given firstname
3. Search for everyone in a college, ORDER BY year, LIMIT 100
4. Search for people with a Facebook ID greater than 10000000000
'''

conn = pymysql.connect(
    db=settings.flexidata_database,
    user=settings.flexidata_username,
    passwd=settings.flexidata_password,
    host=settings.flexidata_host)

types_to_generate = sys.argv[1]
queries_to_make = sys.argv[2]

cursor = conn.cursor(pymysql.cursors.DictCursor)
cursor.execute('SELECT * FROM Students2')

colleges = set()
preferred_names = set()

for row in cursor.fetchall():
    colleges.add(row['college'])
    preferred_names.add(row['preferredName'])

colleges.remove('')
preferred_names.remove('')

if '4' in types_to_generate:
    for i in xrange(1, int(queries_to_make)):
        print 'SELECT netId, firstName, lastName, preferredName, college, classYear, facebookId FROM Students2 WHERE facebookId > 10000000000'