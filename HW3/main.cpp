
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "linalg.h"

namespace hw3
{
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
        operator[](std::string const& key)
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

    class retrieval_accuracy
    {
        inline retrieval_accuracy(std::string const& path)
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
    public:
        word_corpus _words;
        retrieval_accuracy _accuracy;
        Matrix _doc_matrix;

        size_t count = 0;

        inline std::vector<std::string>
        format_string(std::string const& str)
        {
            auto _str = str;
            std::transform(str.begin(), str.end(), _str.begin(), ::tolower);
            auto splitted = std::vector<std::string>(1, _str);

            if(_str.back() == 's')
                splitted.emplace_back(_str.begin(), _str.end() - 1);

            if(std::all_of(_str.begin(),_str.end(), ::isalnum))
                return splitted;
            else
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
                return splitted;
            }
        }

        inline void
        parse_document(std::ifstream& file_stream, size_t doc_idx)
        {
            std::string word;
            while(file_stream >> word)
            {
                auto tokens = format_string(word);

                std::for_each(
                    tokens.begin(), tokens.end(),
                    [&](std::string const& word)
                    {
                        auto word_entry = _words[word];
                        if(word_entry)
                        {
                            ++_doc_matrix(doc_idx, word_entry.value());
                            ++count;
                        }
                    });
            }
        }

    public:
        inline explicit
        document_database(std::string const& path,
                          size_t begin_idx,
                          size_t end_idx)
            : _words(path),
              _accuracy(path),
              _doc_matrix(end_idx - begin_idx, _words.size())
        {
            auto placeholder = std::string();
            auto file_name = path + "/doc000.txt"s;

            std::cout << "-- constructing document matrix\n";
            auto begin = &file_name[file_name.size() - 7];
            for(size_t i = begin_idx; i < end_idx; ++i)
            {
                std::ifstream stream;
                placeholder = std::to_string(10000 + i);
                std::copy(&placeholder[2], &placeholder[5], begin);
                stream.open(file_name);
                parse_document(stream, i);
            }

            std::cout << "-- constructing document matrix - done\n";
            std::cout << "   processed " << end_idx - begin_idx << " documents"
                      << ", counted " << count << " words"
                      << std::endl;
        }
    };
}


int main()
{
    std::cout << "****  CSEG347 HW3: Document Search Engine  ****" << std::endl;
    std::string const path = "database";
    auto db = hw3::document_database<linalg::dense_matrix>(path, 0, 219);

    while(true)
    {
        std::string word;
        std::cout<< "$ ";
        std::cin >> word;
        auto result = db._words[word];
        if(!result)
            std::cout << "not found" << std::endl;
        else
        {
            auto& mat = db._doc_matrix;
            size_t idx = result.value();
            size_t count = std::accumulate(&mat(0, idx),
                                        &mat(0, idx + 1),
                                        0.0f);
            std::cout << "index: " << idx << " count: " << count << std::endl;
        }
    }

    // while(true)
    // {
    //     std::cout<< "$ ";
    //     size_t i, j
    //     std::cin >> i;
    //     std::cin >> j;
    //     auto result = db._words[word];
    //     if(!result)
    //         std::cout << "not found" << std::endl;
    //     else
    //     {
    //         auto& mat = db._doc_matrix;
    //         std::cout << " val: " << mat(i, j) << std::endl;
    //     }
    // }

}
