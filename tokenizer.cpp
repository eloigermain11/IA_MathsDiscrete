#include <unordered_map>
#include <vector>


std::vector<int> load_training_data_set(){

    std::vector<int> initial_tokens;

// READ  from file here

    return initial_tokens;

}


long long pair_to_long_long(int tok1, int tok2) {
    return ((long long)tok1 << 32) | tok2; // transforme le int du tok1 en long long, shift la patie int  de 32 bit cuz est elle centrer 0000..1337...0000000 --> 1337000000...00.  Ajoute le deuxieme int au long long -> 13370000...1887000...00
}




void main() {


int vocab_count = 2000;
int x = 0;

std::unordered_map<long long, int> frequency_count;
std::vector<int> initial_tokens = load_training_data_set();


/*

Read the file into a std::vector<int> — each byte becomes one int (0-255).

Build the pair count map — loop through the vector, for each i, pack (vec[i], vec[i+1]) into a long long, increment it in your unordered_map.

Find the most frequent pair — scan the map for the highest count.

Merge — walk the vector, everywhere you see that pair adjacent:

Decrement old neighbor pairs from the map
Replace the two tokens with a new token ID (starting at 256)
Increment new neighbor pairs in the map
Repeat 3-4 until you hit your target vocab size.

Save the merge rules in order.
*/

while (x < vocab_count) {

    for (var byte of initial_tokens){
    long long key = pair_to_long_long(tok1, tok2);
    frequency_count[key] ++;
    }


    x++;
}



}




