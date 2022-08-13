mosesdecoder=your-path/fairseq


decodes_file=$1
tok_gold_targets=$2
# Replace unicode.
# perl $mosesdecoder/scripts/tokenizer/replace-unicode-punctuation.perl -l de  < $decodes_file > $decodes_file.n

# Tokenize.
# perl $mosesdecoder/scripts/tokenizer/tokenizer.perl -l de < $decodes_file.n > $decodes_file.tok

# Put compounds in ATAT format (comparable to papers like GNMT, ConvS2S).
# See https://nlp.stanford.edu/projects/nmt/ :
# 'Also, for historical reasons, we split compound words, e.g.,
#    "rich-text format" --> rich ##AT##-##AT## text format."'
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < $tok_gold_targets > $tok_gold_targets.atat
perl -ple 's{(\S)-(\S)}{$1 ##AT##-##AT## $2}g' < $decodes_file > $decodes_file.tok.atat

# Get BLEU.
perl $mosesdecoder/multi-bleu.perl $tok_gold_targets.atat < $decodes_file.tok.atat
