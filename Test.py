from flexidata import *

conn = Connection(original_conn)
cur = conn.cursor()

stmt = psqle.parse("UPDATE test_table SET cash=394 WHERE (name='George Bush') AND college='Davenport'")[0]
# #insert_replace_table_name(stmt, 'blah_table')
# stmt = psqle.parse("INSERT INTO test_table (sid, name, college, cash) VALUES"
#                       "(10210101, 'George Bush', 'Davenport', 9999999.54)")[0]
#print_token_children(stmt)
#stmt = psqle.parse("SELECT id AS tableId, uid, table.field, `blah` FROM table1 ORDER BY id ASC, uid DESC GROUP BY id DESC")[0]
print_token_children(stmt)
where = stmt.token_next_by_instance(0, psql.Where)
#print generate_propagate_sql('test_table__0', 'test_table', conn.schemas['test_table'], 'sid', where)
cur.execute("DROP TABLE IF EXISTS test_table, test_table__0, test_table__1, test_table__2")
conn.commit()
cur.execute("INSERT INTO test_table (sid, name, college, cash) VALUES"
            "(10210101, 'George Bush', 'Davenport', 9999999.54)")
conn.commit()
cur.execute("INSERT INTO test_table (sid, name, college, cash, class_year) VALUES "
            "(909876541,'Peter Xu', 'Saybrook', 12.34, '2014')")
conn.commit()
cur.execute("INSERT INTO test_table (sid, name, college, cash, class_year) VALUES "
             "(909876542, 'Peter Xu', 'Morse', 'no mo money yo', '2014')")
conn.commit()
cur.execute("SELECT sid, name, college, cash FROM test_table WHERE cash = 12.34")


# UPDATE test code

stmt = psqle.parse("UPDATE test_table SET cash=394 WHERE (name='George Bush') AND college='Davenport'")[0]
# #insert_replace_table_name(stmt, 'blah_table')
# stmt = psqle.parse("INSERT INTO test_table (sid, name, college, cash) VALUES"
#                       "(10210101, 'George Bush', 'Davenport', 9999999.54)")[0]
#print_token_children(stmt)
#stmt = psqle.parse("SELECT id AS tableId, uid, table.field, `blah` FROM table1 ORDER BY id ASC, uid DESC GROUP BY id DESC")[0]
print_token_children(stmt)
where = stmt.token_next_by_instance(0, psql.Where)
print generate_propagate_sql('test_table__0', 'test_table', conn.schemas['test_table'], 'sid', where)

# cur.execute("INSERT INTO test_table (sid, name, college, cash) VALUES"
#             "(10210101, 'George Bush', 'Davenport', 9999999.54)")
# conn.commit()
# cur.execute("INSERT INTO test_table (sid, name, college, cash, class_year) VALUES "
#             "(909876541,'Peter Xu', 'Saybrook', 12.34, '2014')")
# conn.commit()
# cur.execute("INSERT INTO test_table (id, name, college, cash, class_year) VALUES "
#             "(909876542, 'Peter Xu', 'Morse', 'no mo money yo', '2014')")
# conn.commit()

