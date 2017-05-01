from nltk.parse.stanford import StanfordDependencyParser

# Created by Dulanjaya Tennekoon
# Returns the head word of a given sentence.


# Defining the paths for external jars of the standford dependency parser
__path_to_jar__ = '../../lib_external/stanford_parser/stanford-parser.jar'
__path_to_models_jar__ = '../../lib_external/stanford_parser/stanford-parser-3.7.0-models.jar'

# Standford Dependency Parser
__dp__ = StanfordDependencyParser(path_to_jar=__path_to_jar__, path_to_models_jar=__path_to_models_jar__)


# Generate headword from the SDP
# Example: generate_head_word('I like this restaurant more than others')
# Returns: like
def generate_head_word(sentence):
    result = __dp__.raw_parse(sentence)
    for parse in result:
        return parse.tree()._label
