<?php
// This is easier because it returns the columns in sequence
$mysqli = new mysqli('localhost', 'phsource', 'Im1goodWebadmin!', 'yaleplus');
$result = $mysqli->query('SELECT * FROM Students2');

while ($row = $result->fetch_assoc()) {
	$escapedValues = array();
	foreach ($row as $key => $value) {
		$escapedValues[] = "'" . $mysqli->escape_string($value) . "'";
	}
	echo 'INSERT INTO Students2 (' . implode(', ', array_keys($row)) . ') VALUES (' . implode(', ', $escapedValues) . ")\n";
}