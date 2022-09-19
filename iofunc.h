#pragma once

#include "nn.h"


template<typename T = DEFAULT_SCALAR>
struct IOFunc_ {
	static_assert(std::is_arithmetic<T>::value, "Template parameter T for struct IOFunc_ must be arithmetic");
public:
	using VecR = Eigen::RowVectorX<T>;

	inline static constexpr T max_coeff{ 10 };

	IOFunc_() = default;
	inline IOFunc_(size_t i, size_t o) { this->gen(i, o); }

	void gen(size_t i, size_t o);
	bool execute(VecR& i, VecR& o) const;
	inline bool operator()(VecR& i, VecR& o) const { return this->execute(i, o); }
	void serializeFunc(std::ostream&) const;
	void serializeStructure(std::ostream&) const;

	inline size_t inputs() const { return this->in; }
	inline size_t outputs() const { return this->out; }

protected:
	inline bool checkSize(const VecR& i, const VecR& o) const {
		return (i.cols() == this->in && o.cols() == this->out) || this->in == 0 || this->out == 0;
	}

private:
	size_t in{0}, out{0};
	std::vector<std::array<size_t, 3> > insert_loc;		// { {Idx, Row, Col}, ... }
	mutable Eigen::RowVectorX<T> a;
	mutable Eigen::MatrixX<T> b;

};
typedef IOFunc_		IOFunc;

template<typename nn_t, typename io_t>
inline bool compatible(NeuralNetwork<nn_t>& nn, IOFunc_<io_t>& f)
	{ return nn.inputs() == f.inputs() && nn.outputs() == f.outputs(); }
template<typename nn_t, typename io_t>
inline void genFunc(NeuralNetwork<nn_t>& nn, IOFunc_<io_t>& f)
	{ f.gen(nn.inputs(), nn.outputs()); }
template<typename nn_t, typename ds_t, typename io_t>
inline void genFuncData(NeuralNetwork<nn_t>& nn, DataSet_<ds_t>& d, size_t s, IOFunc_<io_t>& f = IOFunc_<io_t>{})
	{ genFunc(nn, f); genData(nn, d, s, f); }
template<typename nn_t, typename ds_t, typename io_t>
void genData(NeuralNetwork<nn_t>& nn, DataSet_<ds_t>& d, size_t s, IOFunc_<io_t>& f) {
	if (compatible(nn, f)) {
		d.first.clear();
		d.first.reserve(s);
		d.second.clear();
		d.second.reserve(s);
		size_t in = nn.inputs(), out = nn.outputs();
		for (size_t i = 0; i < s; i++) {
			d.first.emplace_back(std::make_unique<VecR_<ds_t> >(in));
			d.second.emplace_back(std::make_unique<VecR_<ds_t> >(out));
			d.first.back()->setRandom();
			if (!f(*d.first.back(), *d.second.back())) {
				std::cout << "Error generating dataset: idx[" << i << "]\n";
			}
		}
	} else {
		std::cout << "Incompatible function IO size for generating dataset\n";
	}
}




template<typename T>
inline T randomRange(T l, T h) { return (rand() / (T)RAND_MAX) * (h - l) + l; }


template<typename s>
void IOFunc_<s>::gen(size_t i, size_t o) {
	// or generate random between i and [?] to use like size i >>
	this->a.resize(i);
	this->a.setRandom();
	this->b.resize(i, o);
	this->b.setRandom();
	for (auto& z : this->a) {
		z = (int)(z * this->max_coeff);
	}
	for (auto& z : this->b.reshaped()) {
		z = (int)(z * this->max_coeff);
	}

	this->insert_loc.resize(i);
	size_t x = rand();
	for (size_t k = 0; k < i; k++) {
		this->insert_loc[k][0] = x++ % 2;
		int r, c;
		if (this->insert_loc[k][0] == 0) {
			r = this->a.rows();
			c = this->a.cols();
		} else {
			r = this->b.rows();
			c = this->b.cols();
		}
		this->insert_loc[k][1] = randomRange<float>(0, r);
		this->insert_loc[k][2] = randomRange<float>(0, c);
	}

	this->in = i;
	this->out = o;

}
template<typename s>
bool IOFunc_<s>::execute(VecR& i, VecR& o) const {
	if (!this->checkSize(i, o)) {
		return false;
	}
	for (size_t k = 0; k < this->in; k++) {
		if (this->insert_loc[k][0] == 0) {
			this->a(this->insert_loc[k][2]) = i[k];
		} else {
			this->b(this->insert_loc[k][0], this->insert_loc[k][2]) = i[k];
		}
	}
	o = this->a * this->b;
	return true;
}
template<typename s>
void IOFunc_<s>::serializeFunc(std::ostream& o) const {
	for (size_t i = 0; i < this->out; i++) {	// for each output ~~ number of cols in b
		o << i + 1 << ". ";
		for (size_t c = 0; c < this->a.cols(); c++) {	// for each part ~~ number of cols in a ~~ number of rows in b
			s e1 = this->a[c];
			s e2 = this->b(c, i);
			char ec1{ '\0' }, ec2{ '\0' };
			for (size_t l = 0; l < this->in; l++) {
				if (this->insert_loc[l][0] == 0 && this->insert_loc[l][2] == c) {
					ec1 = 'a' + l;
				} else if (this->insert_loc[l][1] == c && this->insert_loc[l][2] == i) {
					ec2 = 'a' + l;
				}
			}
			o << "(";
			if (ec1 == '\0') { o << e1; }
			else { o << ec1; }
			o << " * ";
			if (ec2 == '\0') { o << e2; }
			else { o << ec2; }
			o << ") " << (c != this->a.cols() - 1 ? "+ " : "\n");
		}
		o.flush();
	}
	o << std::endl;
}
template<typename s>
void IOFunc_<s>::serializeStructure(std::ostream& o) const {
	o << "[";
	for (size_t i = 0; i < this->in; i++) {
		for (size_t k = 0; k < this->insert_loc.size(); k++) {
			if (this->insert_loc[k][0] == 0 && this->insert_loc[k][2] == i) {
				o << (char)('a' + k);
				goto cont;
			}
		}
		o << this->a[i];
	cont:
		if (i != this->in - 1) {
			o << ", ";
		}
	}
	o << "]\n[\n";
	for (size_t r = 0; r < this->in; r++) {
		o << '\t';
		for (size_t c = 0; c < this->out; c++) {
			for (size_t k = 0; k < this->insert_loc.size(); k++) {
				if (this->insert_loc[k][0] == 1 && this->insert_loc[k][1] == r && this->insert_loc[k][2] == c) {
					o << (char)('a' + k);
					goto cont2;
				}
			}
			o << this->b(r, c);
		cont2:
			if (c != this->out - 1 || r != this->in - 1) {
				o << ", ";
			}
		}
		o << '\n';
	}
	o << "]\n\n";
	o.flush();
}