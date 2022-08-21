#include<stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <semaphore.h>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <algorithm>
#include<iostream>
#include<fstream>
#include<queue>

#pragma comment(lib,"pthreadVC2.lib")
#define MAX_STRING 1000
#define MAX_THREADS 100

using namespace std;

double sigmoid(double x) {
	return 1.0 / exp(1.0 + exp(-x));
}

double inv_sigmoid(double x)
{
	return -log(1.0 / x - 1.0);
}

int min(int a, int b)
{
	if (a < b) return a;
	return b;
}

struct Tuple
{
	int max_len;
	int e[10];
	int r;
	char type;
	int valid;
	double truth, logit;
	std::vector<int> rule_ids;

	Tuple()
	{
		max_len = 0;
		for (int i = 0; i < 10; i++) {
			e[i] = -1;
		}

		r = -1;
		type = -1;
		valid = -1;
		truth = 0;
		logit = 0;
		rule_ids.clear();
	}

	~Tuple() {
		rule_ids.clear();
	}

	void init() {
		truth = 0;
		logit = 0;
		rule_ids.clear();
	}

	friend bool operator < (Tuple u, Tuple v) {
		if (u.max_len < v.max_len) {
			return true;
		}
		else if (u.max_len > v.max_len) {
			return false;
		}
		else {
			if (u.r == v.r) {
				int i = 0;
				while (i < u.max_len) {
					if (u.e[i] < v.e[i]) {
						return true;
					}
					else if (u.e[i] > v.e[i]) {
						return false;
					}
					else {
						i++;
					}
				}
				return false;
			}
			else {
				return u.r < v.r;
			}
		}


	}

	friend bool operator == (Tuple u, Tuple v) {
		if (u.max_len != v.max_len) {
			return false;
		}
		else {
			if (u.r == v.r) {
				int i = 0;
				while (i < u.max_len) {
					if (u.e[i] != v.e[i]) {
						return false;
					}
					i++;
				}
				return true;
			}
			else {
				return false;
			}
		}
	}

};

struct Pair
{
	int e[10];
};


struct Rule
{
	std::vector<int> r_premise;
	int r_hypothesis;
	std::string type;
	double precision, weight, grad;

	Rule() {
		precision = 0;
		weight = 0;
		grad = 0;
	}

	friend bool operator < (Rule u, Rule v) {
		if (u.type == v.type) {
			if (u.r_hypothesis == v.r_hypothesis) {
				int min_length = min(int(u.r_premise.size()), int(v.r_premise.size()));
				for (int k = 0; k != min_length; k++) {
					if (u.r_premise[k] != v.r_premise[k]) {
						return u.r_premise[k] < v.r_premise[k];
					}
				}
			}
			return u.r_hypothesis < v.r_hypothesis;
		}
		return u.type < v.type;
	}

};
char observed_tuple_file[MAX_STRING], probability_file[MAX_STRING], load_file[MAX_STRING], output_rule_file[MAX_STRING], output_prediction_file[MAX_STRING], output_hidden_file[MAX_STRING], save_file[MAX_STRING];
int entity_size = 0, relation_size = 0, tuples_size = 0, observed_tuple_size = 0, hidden_tuple_size = 0, rule_size = 0, iterations = 0, num_threads = 8;
double rule_threshold = 0, tuple_threshold = 0, learning_rate = 0.0001;
long long total_count = 0;
std::map<std::string, int> ent2id, rel2id;
std::vector<std::string> id2ent, id2rel;
std::vector<Tuple> tuples;
std::vector<Pair> *r2e = NULL;
std::set<Rule> candidate_rules;
std::vector<Rule> rules;
std::set<Tuple> observed_tuples, hidden_tuples;
std::map<Tuple, double> tuple2prob;
std::map<Tuple, int> tuple2id;
std::vector<int> rand_idx;
sem_t mutex;



/* Debug */
void print_rule(Rule rule)
{
	for (int k = 0; k != int(rule.r_premise.size()); k++)  cout << id2rel[rule.r_premise[k]].c_str();
	//printf("-> %s %s\n", id2rel[rule.r_hypothesis].c_str(), rule.type.c_str());
	cout << "->" << id2rel[rule.r_hypothesis].c_str() << " " << rule.type.c_str() << endl;
}

/* Debug */
void print_tuple(Tuple tuple)
{

	cout << id2rel[tuple.r] << " ";
	for (int i = 0; i < tuple.max_len; i++) {
		cout << id2ent[tuple.e[i]] << " ";
	}
	cout << endl;
	cout << tuple.type << " " << tuple.valid << " " << tuple.truth << " " << tuple.logit << endl;
	for (int k = 0; k != int(tuple.rule_ids.size()); k++) print_rule(rules[tuple.rule_ids[k]]);
	cout << endl;
	cout << endl;
}

void read_data() {
	//char s_entity[MAX_STRING];
	string s_entity = "";
	string s_relation = "";
	//char s_relation[MAX_STRING];
	int e[10];
	int r;
	Pair rel_ent_pair;
	std::map<std::string, int>::iterator iter;


	ifstream inFile(observed_tuple_file);
	if (!inFile) {
		cout << "Error! file of observed tuples not found!" << endl;
		exit(-1);
	}
	char line[10000];
	char *token;
	const char s[2] = "\t";

	while (inFile.getline(line, 10000)) {

		Tuple tuple;

		char thisline[10000];
		int i = 0;
		strcpy(thisline, line);

		token = strtok(thisline, s);
		s_relation = token;

		if (rel2id.count(s_relation) == 0) {
			rel2id[s_relation] = relation_size;
			id2rel.push_back(s_relation);
			relation_size += 1;
		}
		tuple.r = rel2id[s_relation];

		while (token != NULL) {
			token = strtok(NULL, s);
			if (token != NULL) {
				s_entity = token;
				//cout << "entity: "<<s_entity << endl;
				if (ent2id.count(s_entity) == 0) {
					ent2id[s_entity] = entity_size;
					//cout << "entitiy_size: " << entity_size << endl;
					id2ent.push_back(s_entity);
					entity_size += 1;
				}
				tuple.e[i] = ent2id[s_entity];
				i++;

			}
			tuple.max_len = i;

		}
		tuple.type = 'o';
		tuple.valid = 1;
		tuples.push_back(tuple);
		observed_tuples.insert(tuple);

		//while (token != NULL) {
		//	token = strtok(NULL, s);
		//	s_entity +=  token;
		//}
		//cout << s_entity << " ";
	}
	inFile.close();
	observed_tuple_size = int(tuples.size());

	r2e = new std::vector<Pair>[entity_size];
	for (int k = 0; k != observed_tuple_size; k++) {
		r = tuples[k].r;
		for (int i = 0; i < 10; i++) {
			e[i] = tuples[k].e[i];
			rel_ent_pair.e[i] = e[i];
		}
		r2e[r].push_back(rel_ent_pair);
	}

	cout << "#Entities: " << entity_size << endl;
	cout << "#Relations: " << relation_size << endl;
	cout << "#Observed tuples: " << observed_tuple_size << endl;

}

bool check_observed(Tuple tuple)
{
	if (observed_tuples.count(tuple) != 0) return true;
	else return false;
}

// void search_compositional_rules(int relation, int entities[], int ent_len) {

	// int len1;
	// Rule rule;
	// int flag;

	// int r1_ = 0;
	// while (r1_ < rel2id.size()) {
		// int len_1 = r2e[r1_].size();
		// int e_len_1 = 0;
		// for (int i = 0; i < 10; i++) {
			// if (r2e[r1_][0].e[i] >= 0 and r2e[r1_][0].e[i] < ent2id.size()) {
				// e_len_1++;
			// }
			// else {
				// break;
			// }
		// }
		// for (int k = 0; k < len_1; k++) {
			// if (r2e[r1_][k].e[0] == entities[0]) {

				// int r2_ = 0;
				// while (r2_ < rel2id.size()) {
					// int e_len_2 = 0;
					// for (int i = 0; i < 10; i++) {
						// if (r2e[r2_][0].e[i] >= 0 and r2e[r2_][0].e[i] < ent2id.size()) {
							// e_len_2++;
						// }
						// else {
							// break;
						// }
					// }
					// int len_2 = r2e[r2_].size();
					// for (int k2 = 0; k2 < len_2; k2++) {

						// if (r2e[r2_][k2].e[e_len_2 - 1] == entities[ent_len - 1] and r2e[r2_][k2].e[0] == r2e[r1_][k].e[e_len_1 -1]) {

							// rule.r_premise.clear();
							// rule.r_premise.push_back(r1_);
							// rule.r_premise.push_back(r2_);
							// rule.r_hypothesis = relation;
							// rule.type = "compositional";
							// candidate_rules.insert(rule);

						// }
					// }
					// r2_++;
				// }

			// }
		// }
		// r1_++;
	// }


// }

void search_subrelation_rules(int relation, int entities[], int ent_len)
{
	int len;
	Rule rule;
	int flag = 0;


	int r_ = 0;
	//for (int i = 0; i < 10; i++) {
//	if (r2e[0][0].e[i] != entities[i]) {
//		cout << 1 << endl;
//	}
//}
	while (r_ < rel2id.size()) {
		// 节约时间
		if (r_ == relation) {
			goto label;
		}
		len = r2e[r_].size();

		for (int k = 0; k != len; k++) {
			flag = 0;	// 判断是到结尾了还是遇到的不相同的
			int e_len = 0;
			for (int i = 0; i < 10; i++) {
				if (r2e[r_][k].e[i] >= 0 and r2e[r_][k].e[i] < ent2id.size()) {
					e_len++;
				}
				else {
					break;
				}
			}

			if (e_len != ent_len) {
				flag = 1;
				break;
			}
			for (int i = 0; i < ent_len; i++) {
				if (r2e[r_][k].e[i] != entities[i]) {
					flag = 1;
					break;
				}
			}

			if (flag == 0) {
				rule.r_premise.clear();
				rule.r_premise.push_back(r_);
				rule.r_hypothesis = relation;
				rule.type = "subrelation";
				candidate_rules.insert(rule);
			}

		}
	label:
		r_++;
	}
}

void search_inverse_rules(int relation, int entities[], int ent_len) {

	int len;
	Rule rule;
	int flag = 0;


	int r_ = 0;

	while (r_ < rel2id.size()) {
		// 节约时间
		if (r_ == relation) {
			goto label;
		}
		len = r2e[r_].size();

		for (int k = 0; k != len; k++) {
			flag = 0;	// 判断是到结尾了还是遇到的不相同的
			int e_len = 0;
			for (int i = 0; i < 10; i++) {
				if (r2e[r_][k].e[i] >= 0 and r2e[r_][k].e[i] < ent2id.size()) {
					e_len++;
				}
				else {
					break;
				}
			}

			if (e_len != ent_len) {
				flag = 1;
				break;
			}
			for (int i = 0; i < ent_len; i++) {
				if (r2e[r_][k].e[i] != entities[ent_len - i - 1]) {
					flag = 1;
					break;
				}
			}

			if (flag == 0) {
				rule.r_premise.clear();
				rule.r_premise.push_back(r_);
				rule.r_hypothesis = relation;
				rule.type = "inverse";
				candidate_rules.insert(rule);
			}

		}
	label:
		r_++;
	}

}

void search_symmetric_rules(int relation, int entities[], int ent_len) {

	int len;
	Rule rule;
	int flag = 0;


	int r_ = 0;

	while (r_ < rel2id.size()) {
		// 节约时间
		if (r_ != relation) {
			goto label;
		}
		len = r2e[r_].size();

		for (int k = 0; k != len; k++) {
			flag = 0;	// 判断是到结尾了还是遇到的不相同的
			int e_len = 0;
			for (int i = 0; i < 10; i++) {
				if (r2e[r_][k].e[i] >= 0 and r2e[r_][k].e[i] < ent2id.size()) {
					e_len++;
				}
				else {
					break;
				}
			}

			if (e_len != ent_len) {
				flag = 1;
				break;
			}
			for (int i = 0; i < ent_len; i++) {

				if (r2e[r_][k].e[i] != entities[ent_len - i - 1]) {
					flag = 1;
					break;
				}
			}

			if (flag == 0) {
				rule.r_premise.clear();
				rule.r_premise.push_back(r_);
				rule.r_hypothesis = relation;
				rule.type = "symmetric";
				candidate_rules.insert(rule);
			}

		}
	label:
		r_++;
	}

}



void search_candicate_rules() {
	for (int k = 0; k != observed_tuple_size; k++) {
		if (k % 100 == 0)
		{
			printf("Progress: %.3lf%%          %c", (double)k / (double)(observed_tuple_size + 1) * 100, 13);
			fflush(stdout);
		}
		search_subrelation_rules(tuples[k].r, tuples[k].e, tuples[k].max_len);
		search_inverse_rules(tuples[k].r, tuples[k].e, tuples[k].max_len);
		search_symmetric_rules(tuples[k].r, tuples[k].e, tuples[k].max_len);
		// search_compositional_rules(tuples[k].r, tuples[k].e, tuples[k].max_len);
	}
	std::set<Rule>::iterator iter;
	for (iter = candidate_rules.begin(); iter != candidate_rules.end(); iter++)
		rules.push_back(*iter);

	rule_size = int(candidate_rules.size());
	candidate_rules.clear();
	printf("#Candidate rules: %d          \n", rule_size);

	for (int k = 0; k != rule_size; k++) rand_idx.push_back(k);
	std::random_shuffle(rand_idx.begin(), rand_idx.end());
}

double precision_subrelation_rule(Rule rule) {

	int rp, rh, len;
	double p = 0, q = 0;
	Tuple tuple;
	rp = rule.r_premise[0];
	rh = rule.r_hypothesis;
	len = int(tuples.size());

	for (int k = 0; k != len; k++) {
		if (tuples[k].r != rp) {
			continue;
		}
		for (int i = 0; i < tuples[k].max_len; i++) {
			tuple.e[i] = tuples[k].e[i];
		}
		tuple.r = rh;
		if (check_observed(tuple) == true) {
			p += 1;
		}
		q += 1;

	}

	return p / q;
}

double precision_inverse_rule(Rule rule) {

	int rp, rh, len;
	double p = 0, q = 0;
	Tuple tuple;
	rp = rule.r_premise[0];
	rh = rule.r_hypothesis;
	len = int(tuples.size());

	for (int k = 0; k != len; k++) {
		if (tuples[k].r != rp) {
			continue;
		}
		for (int i = 0; i < tuples[k].max_len; i++) {
			tuple.e[i] = tuples[k].e[tuples[k].max_len - i - 1];
		}
		tuple.r = rh;
		tuple.max_len = tuples[k].max_len;
		if (check_observed(tuple) == true) {
			p += 1;
		}
		q += 1;

	}

	return p / q;
}

double precision_symmetric_rule(Rule rule) {

	int rp, rh, len;
	double p = 0, q = 0;
	Tuple tuple;
	rp = rule.r_premise[0];
	rh = rule.r_hypothesis;
	len = int(tuples.size());

	for (int k = 0; k != len; k++) {
		if (tuples[k].r != rp) {
			continue;
		}
		for (int i = 0; i < tuples[k].max_len; i++) {
			tuple.e[i] = tuples[k].e[tuples[k].max_len - i - 1];
		}
		tuple.r = rh;
		tuple.max_len = tuples[k].max_len;
		if (check_observed(tuple) == true) {
			p += 1;
		}
		q += 1;

	}

	return p / q;
}

void *compute_rule_precision_thread(void *id)
{
	int thread = int((long)(id));
	int bg = int(rule_size / num_threads) * thread;
	int ed = int(rule_size / num_threads) * (thread + 1);
	if (thread == num_threads - 1) ed = rule_size;

	for (int T = bg; T != ed; T++)
	{
		if (T % 10 == 0)
		{
			total_count += 10;
			printf("Progress: %.3lf%%          %c", (double)total_count / (double)(rule_size + 1) * 100, 13);
			fflush(stdout);
		}

		int k = rand_idx[T];

		if (rules[k].type == "subrelation") rules[k].precision = precision_subrelation_rule(rules[k]);
		if (rules[k].type == "inverse") rules[k].precision = precision_inverse_rule(rules[k]);
		if (rules[k].type == "symmetric") rules[k].precision = precision_symmetric_rule(rules[k]);
	}

	pthread_exit(NULL);

}

void compute_rule_precision()
{
	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	total_count = 0;
	for (long a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, compute_rule_precision_thread, (void *)a);
	for (long a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
	free(pt);

	std::vector<Rule> rules_copy(rules);
	rules.clear();
	for (int k = 0; k != rule_size; k++)
	{
		if (rules_copy[k].precision >= rule_threshold) rules.push_back(rules_copy[k]);
	}
	rules_copy.clear();

	rule_size = int(rules.size());
	printf("#Final Rules: %d          \n", rule_size);
}



void search_hidden_with_inverse(int id, int thread) {
	int rp, rh, len;
	int e[10];
	rp = rules[id].r_premise[0];
	rh = rules[id].r_hypothesis;
	len = int(tuples.size());

	for (int k = 0; k != len; k++) {

		Tuple tuple;

		if (tuples[k].r != rp) {
			continue;
		}
		for (int i = 0; i < tuples[k].max_len; i++) {
			tuple.e[i] = tuples[k].e[tuples[k].max_len - i - 1];
		}
		tuple.max_len = tuples[k].max_len;
		tuple.r = rh;
		if (check_observed(tuple) == true) {
			continue;
		}
		tuple.type = 'h';
		tuple.valid = 0;
		sem_wait(&mutex);
		hidden_tuples.insert(tuple);
		sem_post(&mutex);

	}

}

void search_hidden_with_symmetric(int id, int thread) {
	int rp, rh, len;
	int e[10];
	rp = rules[id].r_premise[0];
	rh = rules[id].r_hypothesis;
	len = int(tuples.size());

	for (int k = 0; k != len; k++) {

		Tuple tuple;

		if (tuples[k].r != rp) {
			continue;
		}
		for (int i = 0; i < tuples[k].max_len; i++) {
			tuple.e[i] = tuples[k].e[tuples[k].max_len - i - 1];
		}
		tuple.max_len = tuples[k].max_len;
		tuple.r = rh;
		if (check_observed(tuple) == true) {
			continue;
		}
		tuple.type = 'h';
		tuple.valid = 0;
		sem_wait(&mutex);
		hidden_tuples.insert(tuple);
		sem_post(&mutex);

	}

}

void search_hidden_with_subrelation(int id, int thread) {
	int rp, rh, len;
	int e[10];
	rp = rules[id].r_premise[0];
	rh = rules[id].r_hypothesis;
	len = int(tuples.size());

	for (int k = 0; k != len; k++) {

		Tuple tuple;

		if (tuples[k].r != rp) {
			continue;
		}
		for (int i = 0; i < tuples[k].max_len; i++) {
			tuple.e[i] = tuples[k].e[i];
		}
		tuple.max_len = tuples[k].max_len;
		tuple.r = rh;
		if (check_observed(tuple) == true) {
			continue;
		}
		tuple.type = 'h';
		tuple.valid = 0;
		sem_wait(&mutex);
		hidden_tuples.insert(tuple);
		sem_post(&mutex);

	}

}

void *search_hidden_tuples_thread(void *id)
{
	int thread = int((long)(id));
	int bg = int(rule_size / num_threads) * thread;
	int ed = int(rule_size / num_threads) * (thread + 1);
	if (thread == num_threads - 1) ed = rule_size;

	for (int k = bg; k != ed; k++)
	{
		if (k % 10 == 0)
		{
			total_count += 10;
			printf("Progress: %.3lf%%          %c", (double)total_count / (double)(rule_size + 1) * 100, 13);
			fflush(stdout);
		}

		if (rules[k].type == "subrelation") search_hidden_with_subrelation(k, thread);
		if (rules[k].type == "inverse") search_hidden_with_inverse(k, thread);
		if (rules[k].type == "symmetric") search_hidden_with_symmetric(k, thread);
	}

	pthread_exit(NULL);
}


void search_hidden_tuples()
{
	sem_init(&mutex, 0, 1);
	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	total_count = 0;
	for (long a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, search_hidden_tuples_thread, (void *)a);
	for (long a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
	free(pt);

	hidden_tuple_size = int(hidden_tuples.size());
	int all = observed_tuple_size + hidden_tuple_size;
	tuples_size = all;

	std::set<Tuple>::iterator iter;
	for (iter = hidden_tuples.begin(); iter != hidden_tuples.end(); iter++) tuples.push_back(*iter);
	printf("#Hidden tuples: %d          \n", hidden_tuple_size);
	printf("#Tuples: %d          \n", tuples_size);
}



void read_probability_of_hidden_tuples() {
	if (probability_file[0] == 0) {
		Pair rel_ent_pair;
		tuple2id.clear();
		for (int k = 0; k != entity_size; k++) {
			r2e[k].clear();
		}
		for (int k = 0; k != tuples_size; k++) {
			tuple2id[tuples[k]] = k;
			if (tuples[k].type == 'o') {
				tuples[k].valid = 1;
				tuples[k].truth = 1;
				for (int i = 0; i < tuples[k].max_len; i++) {
					rel_ent_pair.e[i] = tuples[k].e[i];
				}
				r2e[tuples[k].r].push_back(rel_ent_pair);
			}
			else {
				tuples[k].valid = 0;
				tuples[k].truth = 0;
			}
		}
		return;
	}

	string s_entity = "";
	string s_data = "";
	string s_relation = "";
	string s_prob = "";
	double prob;
	Tuple tuple;

	ifstream inFile(probability_file);
	if (!inFile) {
		cout << "ERROR: probability file not found!" << endl;
		exit(1);
	}
	char line[1000];
	char *token;
	const char s[2] = "\t";

	while (inFile.getline(line, 1000)) {

		char thisline[1000];
		string data[11];
		int i = 0;
		strcpy(thisline, line);
		token = strtok(thisline, s);
		s_relation = token;
		if (rel2id.count(s_relation) == 0) {
			continue;
		}
		tuple.r = rel2id[s_relation];

		while (token != NULL) {
			token = strtok(NULL, s);
			if (token != NULL) {
				s_data = token;
				data[i] = s_data;
				i++;
			}
		}
		for (int k = 0; k < i - 1; k++) {
			s_entity = data[k];
			if (ent2id.count(s_entity) == 0) {
				goto label;
			}
			else {
				tuple.e[k] = ent2id[s_entity];
			}

		}

		tuple.max_len = i - 1;
		s_prob = data[i - 1];
		tuple2prob[tuple] = stod(s_prob);


	label:
		;
	}
	inFile.close();

	for (int k = 0; k != tuples_size; k++) {
		if (tuples[k].type == 'o') {
			tuples[k].truth = 1;
			tuples[k].valid = 1;
			continue;
		}
		if (tuple2prob.count(tuples[k]) != 0 && tuple2prob[tuples[k]] >= tuple_threshold) {
			tuples[k].truth = tuple2prob[tuples[k]];
			tuples[k].valid = 1;
		}
		else
		{
			tuples[k].truth = tuple2prob[tuples[k]];
			tuples[k].valid = 0;
		}

	}
	for (int k = 0; k != relation_size; k++) {
		r2e[k].clear();
	}

	Pair rel_ent_pair;
	int e[10];
	int r;
	for (int k = 0; k != tuples_size; k++) {
		tuple2id[tuples[k]] = k;
		if (tuples[k].valid == 0) {
			continue;
		}
		r = tuples[k].r;
		for (int i = 0; i < tuples[k].max_len; i++) {
			rel_ent_pair.e[i] = tuples[k].e[i];
		}
		r2e[r].push_back(rel_ent_pair);
	}

}

void link_subrelation_rule(int id) {
	int tid, h, t, rp, rh;

	rp = rules[id].r_premise[0];
	rh = rules[id].r_hypothesis;

	for (int k = 0; k != tuples_size; k++)
	{

		if (tuples[k].valid == 0) continue;
		if (tuples[k].r != rp) continue;
		Tuple tuple;
		for (int i = 0; i < tuples[k].max_len; i++) {
			tuple.e[i] = tuples[k].e[i];
			tuple.max_len++;
		}
		tuple.r = rh;


		if (tuple2id.count(tuple) == 0) continue;
		tid = tuple2id[tuple];
		sem_wait(&mutex);
		tuples[tid].rule_ids.push_back(id);

		sem_post(&mutex);
	}
}

void link_inverse_rule(int id) {
	int tid, h, t, rp, rh;


	rp = rules[id].r_premise[0];
	rh = rules[id].r_hypothesis;

	for (int k = 0; k != tuples_size; k++)
	{

		if (tuples[k].valid == 0) continue;
		if (tuples[k].r != rp) continue;
		if(tuples[k].type == 'h'){}
		Tuple tuple;
		for (int i = 0; i < tuples[k].max_len; i++) {
			tuple.e[i] = tuples[k].e[tuples[k].max_len - i - 1];
			tuple.max_len++;
		}
		tuple.r = rh;

		if (tuple2id.count(tuple) == 0) continue;
		tid = tuple2id[tuple];
		sem_wait(&mutex);
		tuples[tid].rule_ids.push_back(id);

		sem_post(&mutex);
	}
}

void link_symmetric_rule(int id) {
	int tid, h, t, rp, rh;


	rp = rules[id].r_premise[0];
	rh = rules[id].r_hypothesis;

	for (int k = 0; k != tuples_size; k++)
	{

		if (tuples[k].valid == 0) continue;
		if (tuples[k].r != rp) continue;
		Tuple tuple;
		for (int i = 0; i < tuples[k].max_len; i++) {
			tuple.e[i] = tuples[k].e[tuples[k].max_len - i - 1];
			tuple.max_len++;
		}
		tuple.r = rh;


		if (tuple2id.count(tuple) == 0) continue;
		tid = tuple2id[tuple];
		sem_wait(&mutex);
		tuples[tid].rule_ids.push_back(id);

		sem_post(&mutex);
	}
}

void *link_rules_thread(void *id)
{
	int thread = int((long)(id));
	int bg = int(rule_size / num_threads) * thread;
	int ed = int(rule_size / num_threads) * (thread + 1);
	if (thread == num_threads - 1) ed = rule_size;

	for (int k = bg; k != ed; k++)
	{
		if (k % 10 == 0)
		{
			total_count += 10;
			printf("Progress: %.3lf%%          %c", (double)total_count / (double)((rule_size + 1) * 100), 13);
			fflush(stdout);
		}

		if (rules[k].type == "subrelation") link_subrelation_rule(k);
		if (rules[k].type == "inverse") link_inverse_rule(k);
		if (rules[k].type == "symmetric") link_symmetric_rule(k);
	}

	pthread_exit(NULL);
}

void link_rules()
{
	sem_init(&mutex, 0, 1);
	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	total_count = 0;
	for (long a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, link_rules_thread, (void *)a);
	for (long a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
	free(pt);

	printf("Data preprocessing done!          \n");
}


void init_weight()
{
	for (int k = 0; k != rule_size; k++)
		rules[k].weight = (rand() / double(RAND_MAX) - 0.5) / 100;
}


double train_epoch(double lr)
{
	double error = 0, cn = 0;

	for (int k = 0; k != rule_size; k++) rules[k].grad = 0;

	for (int k = 0; k != tuples_size; k++)
	{
		int len = int(tuples[k].rule_ids.size());
		if (len == 0) continue;

		if(tuples[k].type == 'h'){}
		tuples[k].logit = 0;
		for (int i = 0; i != len; i++)
		{
			int rule_id = tuples[k].rule_ids[i];
			tuples[k].logit += rules[rule_id].weight / len;
		}

		tuples[k].logit = sigmoid(tuples[k].logit);
		for (int i = 0; i != len; i++)
		{
			int rule_id = tuples[k].rule_ids[i];
			rules[rule_id].grad += (tuples[k].truth - tuples[k].logit) / len;
		}

		error += (tuples[k].truth - tuples[k].logit) * (tuples[k].truth - tuples[k].logit);
		cn += 1;
	}

	for (int k = 0; k != rule_size; k++) rules[k].weight += lr * rules[k].grad;

	return sqrt(error / cn);
}

void output_rules()
{
	if (output_rule_file[0] == 0) return;

	FILE *fo = fopen(output_rule_file, "wb");
	for (int k = 0; k != rule_size; k++)
	{
		std::string type = rules[k].type;
		double weight = rules[k].weight;

		fprintf(fo, "%s\t%s\t", type.c_str(), id2rel[rules[k].r_hypothesis].c_str());
		for (int i = 0; i != int(rules[k].r_premise.size()); i++)
			fprintf(fo, "%s\t", id2rel[rules[k].r_premise[i]].c_str());
		fprintf(fo, "%lf\n", weight);
	}
	fclose(fo);
}

void output_predictions()
{
	ofstream ofs;
	ofs.open(output_prediction_file, ios::out);
	if (output_prediction_file[0] == 0) return;

	for (int k = 0; k != tuples_size; k++)
	{
		if (tuples[k].type == 'o') continue;
		int r = tuples[k].r;
		int e[10];
		for (int i = 0; i < tuples[k].max_len; i++) {
			e[i] = tuples[k].e[i];
		}

		double prob = tuples[k].logit;



		ofs << id2rel[r] << "\t";
		for (int i = 0; i < tuples[k].max_len - 1; i++) {
			ofs << id2ent[e[i]] << "\t";
		}
		ofs << id2ent[e[tuples[k].max_len - 1]] << "\t";
		ofs << prob << endl;

	}
	ofs.close();
}


void output_hidden_tuples()
{
	ofstream ofs;
	ofs.open(output_hidden_file, ios::out);
	if (output_hidden_file[0] == 0) return;


	for (int k = 0; k != tuples_size; k++)
	{
		if (tuples[k].type == 'o') continue;
		int r = tuples[k].r;
		int e[10];
		for (int i = 0; i < tuples[k].max_len; i++) {
			e[i] = tuples[k].e[i];
		}
		ofs << id2rel[r] << "\t";
		for (int i = 0; i < tuples[k].max_len - 1; i++) {
			ofs << id2ent[e[i]] << "\t";
		}
		ofs << id2ent[e[tuples[k].max_len - 1]];
		ofs << endl;
	}
	ofs.close();
}


void save()
{
	if (save_file[0] == 0) return;
	ofstream ofs;
	ofs.open(save_file, ios::out);
	ofs << "Entity_size: " << entity_size << endl;
	for (int k = 0; k < entity_size; k++) {
		ofs << id2ent[k].c_str() << "\t";
	}
	ofs << endl;
	ofs << "Relation_size: " << relation_size << endl;
	for (int k = 0; k < relation_size; k++) {
		ofs << id2rel[k].c_str() << "\t";
	}
	ofs << endl;
	ofs << "Tuple_size: " << tuples_size << endl;
	for (int k = 0; k < tuples_size; k++) {
		int r = tuples[k].r;
		int e[10];
		for (int i = 0; i < tuples[k].max_len; i++) {
			e[i] = tuples[k].e[i];
		}
		char type = tuples[k].type;
		int valid = tuples[k].valid;

		ofs << r << "\t";
		for (int i = 0; i < tuples[k].max_len; i++) {
			ofs << e[i] << "\t";
		}
		ofs << type << "\t" << valid << endl;
	}
	ofs << "Rule_size: " << rule_size << endl;
	for (int k = 0; k < rule_size; k++) {
		string type = rules[k].type;
		double weight = rules[k].weight;
		ofs << type.c_str() << "\t" << rules[k].precision << "\t" << id2rel[rules[k].r_hypothesis].c_str() << "\t" << int(rules[k].r_premise.size()) << "\t";
		for (int i = 0; i != int(rules[k].r_premise.size()); i++) {
			ofs << id2rel[rules[k].r_premise[i]].c_str() << "\t";
		}
		ofs << weight << endl;
	}

	ofs.close();
}

//void load()
//{
//	if (load_file[0] == 0) return;
//	ifstream inFile(load_file);
//	if (!inFile) {
//		printf("ERROR: loading file not found!\n");
//		exit(1);
//	}
//
//	FILE *fi = fopen(load_file, "rb");
//	
//
//	fscanf(fi, "%d", &entity_size);
//	id2ent.clear(); ent2id.clear();
//	int eid; char s_ent[MAX_STRING];
//	for (int k = 0; k != entity_size; k++)
//	{
//		fscanf(fi, "%d %s", &eid, s_ent);
//		id2ent.push_back(s_ent);
//		ent2id[s_ent] = eid;
//	}
//
//	fscanf(fi, "%d", &relation_size);
//	id2rel.clear(); rel2id.clear();
//	int rid; char s_rel[MAX_STRING];
//	for (int k = 0; k != relation_size; k++)
//	{
//		fscanf(fi, "%d %s", &rid, s_rel);
//		id2rel.push_back(s_rel);
//		rel2id[s_rel] = rid;
//	}
//
//	fscanf(fi, "%d", &triplet_size);
//	triplets.clear();
//	observed_triplets.clear();
//	h2rt = new std::vector<Pair>[entity_size];
//	int h, r, t;
//	char t_type, s_head[MAX_STRING], s_tail[MAX_STRING];
//	int valid;
//	Triplet triplet;
//	Pair ent_rel_pair;
//	observed_triplet_size = 0; hidden_triplet_size = 0;
//	for (int k = 0; k != triplet_size; k++)
//	{
//		fscanf(fi, "%s %s %s %c %d\n", s_head, s_rel, s_tail, &t_type, &valid);
//		h = ent2id[s_head]; r = rel2id[s_rel]; t = ent2id[s_tail];
//		triplet.h = h; triplet.r = r; triplet.t = t; triplet.type = t_type; triplet.valid = valid;
//		triplet.rule_ids.clear();
//		triplets.push_back(triplet);
//
//		if (t_type == 'o')
//		{
//			observed_triplets.insert(triplet);
//			observed_triplet_size += 1;
//		}
//		else
//		{
//			hidden_triplet_size += 1;
//		}
//
//		if (valid == 0) continue;
//		ent_rel_pair.e = t;
//		ent_rel_pair.r = r;
//		h2rt[h].push_back(ent_rel_pair);
//	}
//
//	fscanf(fi, "%d", &rule_size);
//	rules.clear();
//	Rule rule;
//	char r_type[MAX_STRING];
//	for (int k = 0; k != rule_size; k++)
//	{
//		int cn;
//		fscanf(fi, "%s %lf %s %d", r_type, &rule.precision, s_rel, &cn);
//		rule.r_hypothesis = rel2id[s_rel];
//		rule.type = r_type;
//		rule.r_premise.clear();
//		for (int i = 0; i != cn; i++)
//		{
//			fscanf(fi, "%s", s_rel);
//			rule.r_premise.push_back(rel2id[s_rel]);
//		}
//		fscanf(fi, "%lf", &rule.weight);
//		rules.push_back(rule);
//	}
//
//	fclose(fi);
//
//	printf("#Entities: %d          \n", entity_size);
//	printf("#Relations: %d          \n", relation_size);
//	printf("#Observed triplets: %d          \n", observed_triplet_size);
//	printf("#Hidden triplets: %d          \n", hidden_triplet_size);
//	printf("#Triplets: %d          \n", triplet_size);
//	printf("#Rules: %d          \n", rule_size);
//}

void train()
{
	if (load_file[0] == 0)
	{
		// Read observed triplets
		read_data();
		// Search for candidate logic rules
		search_candicate_rules();
		// Compute the empirical precision of logic rules and filter out low-precision ones
		compute_rule_precision();
		// Search for hidden triplets with the extracted logic rules
		search_hidden_tuples();
	}
	else
	{
		//load();
	}

	save();
	output_hidden_tuples();

	if (iterations == 0) return;

	// Read the probability of hidden triplets predicted by KGE models
	read_probability_of_hidden_tuples();
	// Link each triplet to logic rules which can extract the triplet
	link_rules();
	// Initialize the weight of logic rules randomly
	init_weight();
	for (int k = 0; k != iterations; k++)
	{
		double error = train_epoch(learning_rate);
		printf("Iteration: %d %lf          \n", k, error);
	}
	output_rules();
	output_predictions();
}


void test02() {
	for (int i = 0; i < rule_size; i++) {
		if (rules[i].type == "subrelation") {
			search_hidden_with_subrelation(i, 1);
		}
	}

	cout << "hidden_tuples_size: " << hidden_tuples.size() << endl;

}

void test03() {
	for (int k = 0; k < ent2id.size(); k++) {
		cout << id2ent[k] << " : " << ent2id[id2ent[k]] << endl;
	}
}

void test04() {
	for (int k = 0; k != tuples_size; k++) {
		cout << "prob: " << tuple2prob[tuples[k]] << endl;
	}
}

void test05() {
	for (int i = 0; i < tuples_size; i++) {
		print_tuple(tuples[i]);
	}
}


int ArgPos(char *str, int argc, char **argv)
{
	int a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a]))
	{
		if (a == argc - 1)
		{
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}

int main(int argc, char **argv)
{
	int i;
	if (argc == 1)
	{
		printf("Options:\n");
		printf("Parameters for training:\n");
		printf("\t-observed <file>\n");
		printf("\t\tFile of observed tuples, one tuple per line, with the format: <r> <e1>,...,<en> .\n");
		printf("\t-probability <file>\n");
		printf("\t\tAnnotation of hidden tuples from KHGE model, one tuple per line, with the format: <r> <e1>,...,<en> <prob>.\n");
		printf("\t-out-rule <file>\n");
		printf("\t\tOutput file of logic rules.\n");
		printf("\t-out-prediction <file>\n");
		printf("\t\tOutput file of predictions on hidden tuples by MLN.\n");
		printf("\t-out-hidden <file>\n");
		printf("\t\tOutput file of discovered hidden tuples.\n");
		printf("\t-save <file>\n");
		printf("\t\tSaving file.\n");
		printf("\t-load <file>\n");
		printf("\t\tLoading file.\n");
		printf("\t-iterations <int>\n");
		printf("\t\tNumber of iterations for training.\n");
		printf("\t-lr <float>\n");
		printf("\t\tLearning rate.\n");
		printf("\t-thresh-rule <float>\n");
		printf("\t\tThreshold for logic rules. Logic rules whose empirical precision is less than the threshold will be filtered out.\n");
		printf("\t-thresh-triplet <float>\n");
		printf("\t\tThreshold for triplets. Hidden tuples whose probability is less than the threshold will be viewed as false ones.\n");
		printf("\t-threads <int>\n");
		printf("\t\tNumber of running threads.\n");
		return 0;
	}
	observed_tuple_file[0] = 0;
	probability_file[0] = 0;
	output_rule_file[0] = 0;
	output_prediction_file[0] = 0;
	output_hidden_file[0] = 0;
	save_file[0] = 0;
	load_file[0] = 0;
	if ((i = ArgPos((char *)"-observed", argc, argv)) > 0) strcpy(observed_tuple_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-probability", argc, argv)) > 0) strcpy(probability_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-out-rule", argc, argv)) > 0) strcpy(output_rule_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-out-prediction", argc, argv)) > 0) strcpy(output_prediction_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-out-hidden", argc, argv)) > 0) strcpy(output_hidden_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-save", argc, argv)) > 0) strcpy(save_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-load", argc, argv)) > 0) strcpy(load_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-iterations", argc, argv)) > 0) iterations = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-lr", argc, argv)) > 0) learning_rate = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-thresh-rule", argc, argv)) > 0) rule_threshold = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-thresh-triplet", argc, argv)) > 0) tuple_threshold = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
	train();
	return 0;
}

// int main() {

	// //int num_threads = 3;
	// strcpy(observed_tuple_file, "./train.txt");
	// ////strcpy_s(probability_file, "3.txt");
	// //read_data();

	// strcpy(output_rule_file, "rule_test.txt");
	// strcpy(output_prediction_file, "prediction_test.txt");
	// strcpy(output_hidden_file, "hidden_test.txt");
	// strcpy(save_file, "save_test.txt");

	// strcpy(probability_file, "./annotation.txt");
	// iterations = 100;
	// train();
	// // read_data();
	// // search_candicate_rules();
	// //compute_rule_precision();
	// //search_hidden_tuples();
	// //read_probability_of_hidden_tuples();
	// //link_rules();
	// //test05();
	// //train();

	// //search_candicate_rules();

	// //compute_rule_precision();

	// //search_hidden_tuples();

	// // test01();
	// // test02();
	// //read_probability_of_hidden_tuples();

	// //link_rules();

	// //test05();
	// //output_rules();
	// //output_predictions();
	// //output_hidden_tuples();
	// //save();

	// // system("pause");
	// return 0;
// }