import java.sql.*;

public class Scoreboard {
    public static void main(String[] args) throws Exception {
        String url = "jdbc:mysql://localhost:3306/werewolf";
        String user = "root";
        String password = "password";

        try (Connection conn = DriverManager.getConnection(url, user, password);
             PreparedStatement ps = conn.prepareStatement(
                     "SELECT team, score FROM leaderboard ORDER BY score DESC LIMIT 10");
             ResultSet rs = ps.executeQuery()) {

            int rank = 1;
            while (rs.next()) {
                String team = rs.getString("team");
                int score = rs.getInt("score");
                System.out.printf("%d. %s - %d\n", rank++, team, score);
            }
        }
    }
}
