
using TextAnalysis

files = readdir(pwd())
files = filter(name -> name[end-2:end] == "txt", files)
println("loaded ", size(files)," files")

files = map(name -> TextAnalysis.FileDocument(name), files)
files = map(file -> TextAnalysis.StringDocument(file), files)

function process_document(file)
    TextAnalysis.prepare!(file, TextAnalysis.strip_punctuation)
    TextAnalysis.prepare!(file, TextAnalysis.strip_whitespace)
    TextAnalysis.prepare!(file, TextAnalysis.strip_articles)
    TextAnalysis.prepare!(file, TextAnalysis.strip_indefinite_articles)
    TextAnalysis.prepare!(file, TextAnalysis.strip_definite_articles)
    TextAnalysis.prepare!(file, TextAnalysis.strip_pronouns)
    TextAnalysis.prepare!(file, TextAnalysis.strip_stopwords)
    TextAnalysis.prepare!(file, TextAnalysis.strip_case)
    TextAnalysis.prepare!(file, TextAnalysis.strip_non_letters)
    TextAnalysis.prepare!(file, TextAnalysis.strip_numbers)
    return file
end

function construct_lexicon(corp)
    TextAnalysis.update_lexicon!(corp)
    return corp
end

files = map(file -> process_document(file), files)
corps = map(doc -> TextAnalysis.Corpus([doc]), files)
corps = map(corp -> construct_lexicon(corp), corps)
lexic = map(corp -> TextAnalysis.lexicon(corp), corps)

dict = map(dict -> sort(collect(dict), rev=true, by=elem->elem[2]), lexic)


for (idx, value) in enumerate(dict)
    println(idx)
    for (idx, j) in enumerate(value)
        if idx > 100
            break
        end
        println(j)
    end
    println("\n\n")
end
