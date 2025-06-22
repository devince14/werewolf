# Scoreboard Service (Java)

This example Java program demonstrates how a scoreboard might query a MySQL
leaderboard table and print the top teams. It assumes a database `werewolf` with
a table `leaderboard(team VARCHAR(255), score INT)` and appropriate privileges.

Compile and run using a JDK and the MySQL JDBC driver:

```bash
javac -cp mysql-connector-j.jar Scoreboard.java
java  -cp .:mysql-connector-j.jar Scoreboard
```

The program simply prints the top 10 teams in descending score order.
