using System;
using System.Linq;
using Microsoft.ML.Probabilistic;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;

namespace myApp
{
    class PlayerRank
    {
        public static void Run()
        {
            // The winner and loser in each of samples games
            var squadProjection = new[] { "乌拉圭", "葡萄牙", "法国", "阿根廷", "巴西", "墨西哥", "比利时", "日本", "西班牙", "俄罗斯", "克罗地亚", "丹麦", "瑞典", "瑞士", "哥伦比亚", "英格兰" };
            // Index                       0        1       2       3       4       5        6         7      8        9        10        11      12     13     14         15      
            var winnerData = new[] { 0, 2, 4, 6, 9, 10, 12, 15, 2, 6, 10, 15, 2, 10, 6 };
            var loserData = new[] { 1, 3, 5, 7, 8, 11, 13, 14, 0, 4, 9, 12, 6, 15, 15 };

            // Define the statistical model as a probabilistic program 
            var game = new Range(winnerData.Length);
            var player = new Range(winnerData.Concat(loserData).Max() + 1);
            var playerSkills = Variable.Array<double>(player);
            playerSkills[player] = Variable.GaussianFromMeanAndVariance(6, 9).ForEach(player);

            var winners = Variable.Array<int>(game);
            var losers = Variable.Array<int>(game);

            using (Variable.ForEach(game))
            {
                // The player performance is a noisy version of their skill
                var winnerPerformance = Variable.GaussianFromMeanAndVariance(playerSkills[winners[game]], 9);
                var loserPerformance = Variable.GaussianFromMeanAndVariance(playerSkills[losers[game]], 9);

                // The winner performed better in this game
                Variable.ConstrainTrue(winnerPerformance > loserPerformance);
            }

            // Attach the data to the model
            winners.ObservedValue = winnerData;
            losers.ObservedValue = loserData;

            // Run inference
            var inferenceEngine = new InferenceEngine();
            inferenceEngine.ShowFactorGraph = true;
            var inferredSkills = inferenceEngine.Infer<Gaussian[]>(playerSkills);

            // The inferred skills are uncertain, which is captured in their variance
            var orderedPlayerSkills = inferredSkills
                .Select((s, i) => new { Player = i, Skill = s })
                .OrderByDescending(ps => ps.Skill.GetMean());

            foreach (var playerSkill in orderedPlayerSkills)
            {
                Console.WriteLine($"Player {playerSkill.Player}: {squadProjection[playerSkill.Player]} skill: {playerSkill.Skill}");
            }
        }
    }
}