#include "pentago/base/board.h"
#include "pentago/base/score.h"
#include "pentago/base/superscore.h"
#include "pentago/base/symmetry.h"
#include "pentago/data/numpy.h"
#include "pentago/utility/array.h"
#include "pentago/utility/const_cast.h"
#include "pentago/utility/memory.h"
#include "pentago/utility/random.h"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <map>
#include <chrono>
#include <algorithm>
#include <unordered_map>

namespace pentago {

using std::cout;
using std::endl;
using std::vector;
using std::string;
using std::map;
using std::unordered_map;
using std::unique_ptr;
using std::make_unique;
using std::exception;
using tinyformat::format;

// Structure to hold board and outcome data
struct sample_t {
  board_t board;
  Vector<super_t,2> wins;  // black wins, white wins
};

// Base class for compression strategies
class CompressionStrategy {
public:
  virtual ~CompressionStrategy() {}

  virtual string name() const = 0;

  // Train model on data
  virtual void train(RawArray<const sample_t> data) = 0;

  // Predict outcome for a board (1 for win, -1 for loss, 0 for tie)
  virtual int predict(board_t board) = 0;

  // Evaluate accuracy on test data
  void evaluate(RawArray<const sample_t> test_data) {
    int correct = 0;
    int total = 0;
    int win_correct = 0;
    int win_total = 0;
    int loss_correct = 0;
    int loss_total = 0;
    int tie_correct = 0;
    int tie_total = 0;

    for (const auto& data : test_data) {
      // Check all 256 possible rotations
      for (const int r : range(256)) {
        // Get actual outcome (1 for black win, -1 for white win, 0 for tie)
        int actual = data.wins[0](r) - data.wins[1](r);

        // Apply rotation to the board using symmetry_t
        const symmetry_t sym(0, r);  // 0 for global rotation, r for local rotations
        board_t rotated = transform_board(sym, data.board);

        // Make prediction
        int predicted = predict(rotated);

        // Update statistics
        total++;
        if (predicted == actual) correct++;

        if (actual == 1) {
          win_total++;
          if (predicted == 1) win_correct++;
        } else if (actual == -1) {
          loss_total++;
          if (predicted == -1) loss_correct++;
        } else {
          tie_total++;
          if (predicted == 0) tie_correct++;
        }
      }
    }

    double overall_accuracy = total > 0 ? 100.0 * correct / total : 0;
    double win_accuracy = win_total > 0 ? 100.0 * win_correct / win_total : 0;
    double loss_accuracy = loss_total > 0 ? 100.0 * loss_correct / loss_total : 0;
    double tie_accuracy = tie_total > 0 ? 100.0 * tie_correct / tie_total : 0;

    cout << "Strategy: " << name() << endl;
    cout << "  Overall accuracy: " << overall_accuracy << "% (" << correct << "/" << total << ")" << endl;
    cout << "  Win accuracy:     " << win_accuracy << "% (" << win_correct << "/" << win_total << ")" << endl;
    cout << "  Loss accuracy:    " << loss_accuracy << "% (" << loss_correct << "/" << loss_total << ")" << endl;
    if (tie_total > 0) {
      cout << "  Tie accuracy:     " << tie_accuracy << "% (" << tie_correct << "/" << tie_total << ")" << endl;
    }
    cout << endl;
  }
};

// Strategy 1: Simple stone counting heuristic
class StoneCountingStrategy : public CompressionStrategy {
public:
  string name() const override {
    return "Stone Counting Heuristic";
  }

  void train(RawArray<const sample_t> data) override {
    // No training needed for this simple heuristic
  }

  int predict(board_t board) override {
    // Extract sides
    side_t side0 = unpack(board, 0); // Black
    side_t side1 = unpack(board, 1); // White

    // Count stones for each player
    int black_count = popcount(side0);
    int white_count = popcount(side1);

    // Use stone count difference as a simple heuristic
    // Positive numbers favor black, negative favor white
    int diff = black_count - white_count;

    // Determine whose turn it is
    bool black_turn = black_to_move(board);

    // Adjust prediction based on whose turn it is
    if (black_turn) {
      return diff > 0 ? 1 : -1;  // Black's turn: black advantage → win
    } else {
      return diff < 0 ? 1 : -1;  // White's turn: white advantage → win
    }
  }
};

// Strategy 2: Pattern-based heuristic using deviation from baseline
class PatternStrategy : public CompressionStrategy {
public:
  string name() const override {
    return "Pattern-Based Heuristic";
  }

  void train(RawArray<const sample_t> data) override {
    // First pass: collect pattern statistics and compute baseline
    int total_wins = 0;
    int total_samples = 0;

    for (const auto& sample : data) {
      // For each rotation
      for (const int r : range(256)) {
        // Apply rotation to the board using symmetry_t
        const symmetry_t sym(0, r);  // 0 for global rotation, r for local rotations
        board_t rotated = transform_board(sym, sample.board);

        // Extract information from the board
        side_t black = unpack(rotated, 0);
        side_t white = unpack(rotated, 1);

        // Determine outcome and update patterns
        int outcome = sample.wins[0](r) - sample.wins[1](r);
        bool win = outcome > 0;

        // Update pattern frequencies
        update_patterns(black, white, win);

        // Track global statistics for baseline
        total_samples++;
        if (win) total_wins++;
      }
    }

    // Calculate baseline win rate
    baseline_win_rate = static_cast<double>(total_wins) / total_samples;
    cout << "  Baseline win rate: " << (baseline_win_rate * 100) << "%" << endl;

    // Calculate pattern deviations from baseline
    const int min_count = 100;  // Require minimum samples for reliable deviation estimate
    int patterns_used = 0;
    for (auto& [pattern, stats] : pattern_stats) {
      if (stats.total >= min_count) {
        double pattern_win_rate = static_cast<double>(stats.wins) / stats.total;
        stats.deviation = pattern_win_rate - baseline_win_rate;
        patterns_used++;
      } else {
        stats.deviation = 0.0;  // Not enough data, assume baseline
      }
    }
    cout << "  Patterns with enough data: " << patterns_used << " / " << pattern_stats.size() << endl;
  }

  int predict(board_t board) override {
    if (pattern_stats.empty()) {
      // Fallback if no patterns learned
      return -1;  // Predict majority class (loss)
    }

    // Extract sides
    side_t black = unpack(board, 0);
    side_t white = unpack(board, 1);

    double score = 0.0;
    int count = 0;

    // Check for patterns in each quadrant
    for (const int q : range(4)) {
      quadrant_t q_black = quadrant(black, q);
      quadrant_t q_white = quadrant(white, q);

      // Create a pattern key using both players' stones in this quadrant
      uint32_t pattern_key = (q_black << 16) | q_white;

      auto it = pattern_stats.find(pattern_key);
      if (it != pattern_stats.end() && it->second.total >= 100) {
        score += it->second.deviation;
        count++;
      }
    }

    if (count == 0) {
      return -1;  // No pattern info, predict majority class
    }

    // Sum of deviations: positive means above baseline (favor win), negative means below
    // Use a threshold to decide; threshold of 0 means: if evidence suggests above baseline, predict win
    const double threshold = 0.0;
    return (score > threshold) ? 1 : (score < -threshold ? -1 : 0);
  }

private:
  struct PatternStats {
    int wins = 0;
    int total = 0;
    double deviation = 0.0;  // Deviation from baseline win rate
  };

  double baseline_win_rate = 0.0;

  // Map from pattern hash to stats (using uint32_t to combine black and white pieces)
  unordered_map<uint32_t, PatternStats> pattern_stats;

  // Update pattern statistics
  void update_patterns(side_t black, side_t white, bool win) {
    // Check patterns in each quadrant
    for (const int q : range(4)) {
      quadrant_t q_black = quadrant(black, q);
      quadrant_t q_white = quadrant(white, q);

      // Create pattern key combining both players' stones
      uint32_t pattern_key = (q_black << 16) | q_white;

      auto& stats = pattern_stats[pattern_key];
      stats.total++;
      if (win) stats.wins++;
    }
  }
};

// Strategy 3: Machine learning approach using feature engineering
class FeatureBasedStrategy : public CompressionStrategy {
public:
  string name() const override {
    return "Feature-Based Machine Learning";
  }

  void train(RawArray<const sample_t> data) override {
    vector<vector<double>> features;
    vector<int> labels;

    // Extract features and labels from data
    for (const auto& sample : data) {
      // For each rotation
      for (const int r : range(256)) {
        // Apply rotation to the board using symmetry_t
        const symmetry_t sym(0, r);  // 0 for global rotation, r for local rotations
        board_t rotated = transform_board(sym, sample.board);

        // Extract features
        vector<double> feat = extract_features(rotated);
        features.push_back(feat);

        // Get label (1 for black win, -1 for white win, 0 for tie)
        labels.push_back(sample.wins[0](r) - sample.wins[1](r));
      }
    }

    // Train a simple linear model
    train_linear_model(features, labels);
  }

  int predict(board_t board) override {
    if (weights.empty()) {
      return 0; // No model trained
    }

    // Extract features
    vector<double> features = extract_features(board);

    // Apply model
    double score = bias;
    for (const int i : range(min(features.size(), weights.size()))) {
      score += features[i] * weights[i];
    }

    // Convert score to prediction
    return score > 0.1 ? 1 : (score < -0.1 ? -1 : 0);
  }

private:
  vector<double> weights;
  double bias = 0.0;

  // Extract features from a board
  vector<double> extract_features(board_t board) {
    vector<double> features;

    // Extract sides
    side_t black = unpack(board, 0);
    side_t white = unpack(board, 1);

    // Feature 1: Number of stones for each side
    int black_count = popcount(black);
    int white_count = popcount(white);
    features.push_back(static_cast<double>(black_count));
    features.push_back(static_cast<double>(white_count));

    // Feature 2: Number of potential winning lines for each side
    // This would require complex line analysis, simplifying for now
    features.push_back(static_cast<double>(black_count));  // Placeholder
    features.push_back(static_cast<double>(white_count));  // Placeholder

    // Feature 3: Control of center positions
    // Convert board to table to check center positions
    Array<int,2> table = to_table(board);
    int center_control = 0;
    for (const int y : range(2, 4)) {
      for (const int x : range(2, 4)) {
        if (table(x,y) == 1) center_control++;
        else if (table(x,y) == 2) center_control--;
      }
    }
    features.push_back(static_cast<double>(center_control));

    // Feature 4: Quadrant stone distributions
    for (const int q : range(4)) {
      quadrant_t black_q = quadrant(black, q);
      quadrant_t white_q = quadrant(white, q);
      features.push_back(static_cast<double>(popcount(black_q)));
      features.push_back(static_cast<double>(popcount(white_q)));
    }

    return features;
  }

  // Train a simple linear model using stochastic gradient descent
  void train_linear_model(const vector<vector<double>>& features, const vector<int>& labels) {
    if (features.empty() || features[0].empty()) return;

    const int num_features = features[0].size();
    weights.resize(num_features, 0.0);
    bias = 0.0;

    // Initialize weights with small random values
    Random random(1234); // Fixed seed for reproducibility
    for (auto& w : weights) {
      w = random.uniform<double>(-0.01, 0.01);
    }

    // Simple stochastic gradient descent
    const double learning_rate = 0.01;
    const int epochs = 100;

    for (const int epoch __attribute__((unused)) : range(epochs)) {
      for (const int i : range(features.size())) {
        // Predict
        double pred = bias;
        for (const int j : range(num_features)) {
          pred += features[i][j] * weights[j];
        }

        // Compute gradient
        double target = static_cast<double>(labels[i]);
        double error = target - pred;

        // Update weights and bias
        bias += learning_rate * error;
        for (const int j : range(num_features)) {
          weights[j] += learning_rate * error * features[i][j];
        }
      }
    }
  }
};

int toplevel(int argc, char** argv) {
  try {
    // Hardcoded path to the sparse-17.npy file (absolute path)
    string npy_path = "/Users/irving/pentago/data/edison/project/all/sparse-17.npy";
    cout << "Loading data from " << npy_path << "..." << endl;

    // Load the numpy data via mmap
    auto raw_data = read_numpy<uint64_t, 9>(npy_path);

    cout << "Loaded " << raw_data.size() << " samples" << endl;

    // Convert to sample_t for easier processing
    Array<sample_t> board_data(raw_data.size());
    for (const int i : range(raw_data.size())) {
      board_data[i].board = board_t(raw_data[i][0]);
      for (const int j : range(2)) {
        const int offset = 1 + 4*j;
        board_data[i].wins[j] = super_t(raw_data[i][offset], raw_data[i][offset+1],
                                        raw_data[i][offset+2], raw_data[i][offset+3]);
      }
    }

    // Print data statistics
    int total_outcomes = 0;
    int win_outcomes = 0;
    int loss_outcomes = 0;
    int tie_outcomes = 0;

    for (const auto& data : board_data) {
      for (const int r : range(256)) {
        total_outcomes++;
        int outcome = data.wins[0](r) - data.wins[1](r);
        if (outcome > 0) {
          win_outcomes++;
        } else if (outcome < 0) {
          loss_outcomes++;
        } else {
          tie_outcomes++;
        }
      }
    }

    cout << "Data statistics:" << endl;
    cout << "  Total boards: " << board_data.size() << endl;
    cout << "  Total known outcomes: " << total_outcomes << endl;
    cout << "  Win outcomes: " << win_outcomes << " (" << (100.0 * win_outcomes / total_outcomes) << "%)" << endl;
    cout << "  Loss outcomes: " << loss_outcomes << " (" << (100.0 * loss_outcomes / total_outcomes) << "%)" << endl;
    if (tie_outcomes > 0) {
      cout << "  Tie outcomes: " << tie_outcomes << " (" << (100.0 * tie_outcomes / total_outcomes) << "%)" << endl;
    }
    cout << endl;

    // Create and evaluate different compression strategies
    vector<unique_ptr<CompressionStrategy>> strategies;
    //strategies.push_back(make_unique<StoneCountingStrategy>());
    strategies.push_back(make_unique<PatternStrategy>());
    //strategies.push_back(make_unique<FeatureBasedStrategy>());

    // Train and evaluate each strategy
    for (auto& strategy : strategies) {
      cout << "Training " << strategy->name() << "..." << endl;
      strategy->train(board_data);

      cout << "Evaluating " << strategy->name() << "..." << endl;
      strategy->evaluate(board_data);
      cout << endl;
    }

    return 0;
  } catch (const exception& e) {
    cout << "Error: " << e.what() << endl;
    return 1;
  }
}

} // namespace pentago

int main(int argc, char** argv) {
  return pentago::toplevel(argc, argv);
}