
#include <chrono>
#include <exception>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <string>
#include <sstream>
#include <vector>

#include "linalg.h"

namespace hw3
{
    inline decltype(auto)
    time_elapsed(std::chrono::time_point<std::chrono::steady_clock> begin,
                 std::chrono::time_point<std::chrono::steady_clock> end)
    {
        return std::chrono::duration_cast<
            std::chrono::milliseconds>(end - begin);
    }

    using namespace std::string_literals;
    class word_corpus
    {
        std::vector<std::string> _words; 

        std::vector<std::string>
        read_words(std::ifstream& word_stream)
        {
            std::vector<std::string> words;
            words.reserve(8000);

            std::string line;
            while(word_stream >> line)
                words.emplace_back(line); 
            words.shrink_to_fit();
            return words;
        }

    public:
        inline explicit
        word_corpus(std::string const& path)
            :_words()
        {
            std::cout << "-- constructing word corpus\n";

            std::ifstream word_stream;
            word_stream.open(path + "/wordlist.txt"s);
            _words = read_words(word_stream);
            
            std::cout << "-- constructing word corpus - done\n";
            std::cout << "   total of " << _words.size() << " words"
                      << std::endl;
        }

        inline size_t
        size() const noexcept
        {
            return _words.size();
        }

        inline std::optional<size_t>
        operator[](std::string const& key) const
        {
            auto elem = std::lower_bound(_words.begin(), _words.end(), key);
            if(elem == _words.end() || key != *elem)
                return {};
            else
                return elem - _words.begin();
        }

        inline std::string const&
        operator[](size_t index)
        {
            return _words[index];
        }
    };

    class optimizer
    {
    public:
        inline optimizer(std::string const& path)
        {
            std::ifstream word_stream;
            word_stream.open(path + "/documentkey.txt"s);
            /* by keys in the document, compute f score whenever asked for */
            /* probably use cross-entropy */
        }
    };

    template<typename Matrix>
    class document_database
    {
    private:
        word_corpus _words;
        optimizer _accuracy;
        Matrix _latent_space_matrix;
        Matrix _doc_matrix;
        Matrix _encoder_matrix;

        size_t count = 0;

        inline std::vector<std::string>
        split_string(std::string const& str) const
        {
            auto _str = str;
            auto splitted = std::vector<std::string>(1, _str);

            if(_str.back() == 's')
                splitted.emplace_back(_str.begin(), _str.end() - 1);

            if(!std::all_of(_str.begin(),_str.end(), ::isalnum))
            {
                auto begin = _str.begin();
                auto end = _str.end();
                auto prev = begin;
                auto curr = std::find_if_not(prev, end, ::isalnum);
                while(curr != end)
                {
                    splitted.emplace_back(prev, curr); 
                    prev = std::find_if(curr+1, end, ::isalnum);
                    curr = std::find_if_not(prev, end, ::isalnum);
                }
                splitted.emplace_back(prev, curr);
            }
            return splitted;
        }

        inline void
        parse_document(std::ifstream& file_stream, size_t doc_idx, bool pred)
        {
            std::vector<std::string> words;
            words.reserve(1000);
            std::copy(std::istream_iterator<std::string>(file_stream),
                      std::istream_iterator<std::string>(), std::back_inserter(words));

            if(words.size() == 0)
                return;

            std::transform(words.begin(), words.end(), words.begin(),
                           [](std::string const& word){
                               std::string result(word);
                               std::transform(word.begin(), word.end(), result.begin(), ::tolower);
                               return result;
                           });

            auto register_word =
                [&](std::string word){
                    auto word_entry = _words[word];
                    if(word_entry)
                    {
                        ++_doc_matrix(doc_idx, word_entry.value());
                        ++count;
                    }
                };

            for(auto it = words.begin(); it != words.end(); ++it)
            {
                auto tokens = split_string(*it);
                std::for_each(tokens.begin(), tokens.end(), register_word);
            }

            for(auto it = words.begin(); std::next(it) != words.end(); ++it)
            {
                register_word(*it + *std::next(it));
                register_word(*it + '_' + *std::next(it));
                register_word(*it + '-' + *std::next(it));
            }
        }

        inline std::vector<double>
        compute_similarity(std::vector<double> const& word_vec) const
        {
            auto query_norm = linalg::norm(word_vec);
            auto temp = gemv(1.0, _encoder_matrix, word_vec);
            return gemv((1 / query_norm), _latent_space_matrix, temp);
        }

        inline std::vector<size_t>
        topk_similar_documents(std::vector<double> const& similarity, size_t k) const
        {
            auto highest_elem_indices = std::vector<size_t>();
            highest_elem_indices.reserve(k + 1);

            for(size_t i = 0; i < k; ++i)
                highest_elem_indices.push_back(i);

            std::sort(highest_elem_indices.begin(),
                      highest_elem_indices.end(),
                      [&similarity](size_t first, size_t second)
                      { return similarity[first] > similarity[second]; });

            for(size_t i = k; i < similarity.size(); ++i)
            {
                if(similarity[i] > similarity[highest_elem_indices.back()])
                {
                    auto it = std::upper_bound(
                        highest_elem_indices.begin(),
                        highest_elem_indices.end(),
                        similarity[i],
                        [&similarity](double val, size_t second)
                        { return val >= similarity[second]; });

                    auto begin = highest_elem_indices.begin();
                    auto dist = it - begin ;
                    highest_elem_indices.insert(begin + dist, i);
                    highest_elem_indices.pop_back();
                }
            }
            return highest_elem_indices;
        }

    public:
        inline explicit
        document_database(std::string const& path,
                          size_t begin_idx,
                          size_t end_idx)
            : _words(path),
              _accuracy(path),
              _doc_matrix(end_idx - begin_idx, _words.size()),
              _encoder_matrix()
        {
            auto placeholder = std::string();
            auto file_name = path + "/doc000.txt"s;
            auto big_file_name = path + "/doc0000.txt"s;

            std::cout << "-- constructing document matrix\n";
            auto start = std::chrono::steady_clock::now();
            for(size_t i = begin_idx; i < end_idx; ++i)
            {
                std::ifstream stream;
                placeholder = std::to_string(10000 + i);
                if(i < 1000)
                {
                    auto begin = &file_name[file_name.size() - 7];
                    std::copy(&placeholder[2], &placeholder[5], begin);
                    stream.open(file_name);
                }
                else
                {
                    auto begin = &big_file_name[big_file_name.size() - 8];
                    std::copy(&placeholder[1], &placeholder[5], begin);
                    stream.open(big_file_name);
                }

                if(stream)
                    parse_document(stream, i, false);
            }
            auto stop = std::chrono::steady_clock::now();

            std::cout << "-- constructing document matrix - done\n";
            std::cout << "   processed " << end_idx - begin_idx << " documents"
                      << ", counted " << count << " words\n"
                      << "   time elapsed is " << time_elapsed(start, stop).count() << "ms"
                      << std::endl;
            _doc_matrix.normalize_cols();
        }

        inline std::vector<std::pair<size_t, double>>
        query(std::vector<std::string> terms,
              size_t num_result) const
        {
            std::vector<double> word_vec(_words.size(), 0);
            for(auto const& i : terms)
            {
                auto entry = _words[i];
                if(!entry)
                    throw std::runtime_error('\'' + i + "\' not found in database.");
                word_vec[entry.value()] = 1;
            }
            auto sim_vec = compute_similarity(word_vec);
            auto top_k_documents = topk_similar_documents(sim_vec, num_result);

            auto results = std::vector<std::pair<size_t, double>>(num_result);
            for(size_t i = 0; i < num_result; ++i)
            {
                size_t doc_idx  = top_k_documents[i];
                results[i].first = doc_idx;
                results[i].second = sim_vec[doc_idx];
            }
            return results; 
        }

        inline void
        decompose(size_t k)
        {
            std::cout << "-- computing low rank approximation \n";
            auto start = std::chrono::steady_clock::now();

            auto [U, D, Vt] = svd(_doc_matrix);       
            _latent_space_matrix = approx_matscaling(U, D, k);
            _encoder_matrix = std::move(Vt);

            auto norms = std::vector<double>(_latent_space_matrix.shape().first);
            for(size_t i = 0; i < _doc_matrix.shape().first; ++i)
                norms[i] = _latent_space_matrix.row_norm(i);

            for(size_t j = 0; j < _latent_space_matrix.shape().second; ++j) {
                for(size_t i = 0; i < _latent_space_matrix.shape().first; ++i)
                    _doc_matrix(i, j) /= norms[i];
            }
            auto stop = std::chrono::steady_clock::now();
            std::cout << "-- computing low rank approximation - done\n"
                      << "   time elapsed is " << time_elapsed(start, stop).count() << "ms"
                      << std::endl;
        }
    };

    class terminal
    {
    private:
        document_database<linalg::dense_matrix> _db;

    public:
        explicit terminal(std::string const& path)
            : _db(path, 0, 1012)
        {
            size_t default_k = 500;
            std::cout << "-- using default approximation rank\n"
                      << "   default rank is " << default_k
                      << std::endl;
            _db.decompose(default_k);
        }

        void
        query(std::string const& query) const
        {
            std::string term;
            std::stringstream stream(query);
            std::vector<std::string> terms{};
            while(stream >> term)
                terms.push_back(std::move(term));

            auto result = std::vector<std::pair<size_t, double>>();
            auto start = std::chrono::steady_clock::now();
            try
            { result = _db.query(terms,10); }
            catch(std::runtime_error const& e)
            {
                std::cout << " Error: " << e.what() << std::endl;
                return;
            }
            auto stop = std::chrono::steady_clock::now();


            std::cout << "-----   query result   -----\n\n"
                      << "     document  cosine \n"
                      << "     index     similarity\n"
                      << std::endl;
            for(size_t i = 0; i < result.size(); ++i)
            {
                printf("%3d %4d       %f\n",
                       (int)i+1, (int)result[i].first, result[i].second);
            }
            std::cout << "\n time elapsed is "
                      << time_elapsed(start, stop).count() << "ms\n"
                      << std::endl;
        }
    };
}

int main()
{
    std::cout << "****  CSEG347 HW3: Document Search Engine  ****" << std::endl;
    std::string const path = "database";
    hw3::terminal _term(path);

    while(true)
    {
        std::string input;
        std::cout<< "$ ";
        std::getline(std::cin, input);
        if(input.empty())
            continue;
        if(input[0] != ':')
            _term.query(input);
    }
    // auto& mat = db._doc_matrix;
    // auto& corpus = db._words;
    // for(size_t i = 0; i < mat.shape().second; ++i)
    // {
    //     double count = std::accumulate(&mat(0, i),
    //                                    &mat(0, i + 1),
    //                                    0.0);
    //     if(count == 0)
    //     {
    //         std::cout << "found no occurance of: \"" << corpus[i]
    //                   << "\" index: " << i << std::endl;
    //     }
    // }

    // while(true)
    // {
    //     std::string word;
    //     std::cout<< "$ ";
    //     std::cin >> word;
    //     auto result = db._words[word];
    //     if(!result)
    //         std::cout << "not found" << std::endl;
    //     else
    //     {
    //         size_t idx = result.value();
    //         size_t count = std::accumulate(&mat(0, idx),
    //                                        &mat(0, idx + 1),
    //                                        0.0f);
    //         std::cout << "index: " << idx << " count: " << count << std::endl;
    //     }
    // }
}
