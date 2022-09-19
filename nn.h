#pragma once

#include <type_traits>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>
#include <array>

#include "AllEigen"


#define DEFAULT_SCALAR	float

template<typename T> using Matrix_		=	Eigen::Matrix<T, -1, -1, Eigen::RowMajor>;
template<typename T> using VecR_		=	Eigen::RowVectorX<T>;
template<typename T> using VecC_		=	Eigen::VectorX<T>;

template<typename T> using IOList_		=	std::vector<std::unique_ptr<VecR_<T> > >;
template<typename T> using DataSet_		=	std::pair<IOList_<T>, IOList<T> >;
template<typename T> using DataPoint_	=	std::pair<std::unique_ptr<VecR<T> >, std::unique_ptr<VecR<T> > >;
template<typename T> using DataSetR_	=	std::vector<DataPoint_<T> >;


enum ActivationFunc {
	LINEAR = 0,
	RELU,
	SIGMOID,
	TANH,
	SINE,
	COSINE
};
enum Regularization {
	NONE = 0,
	L1,
	L2,
	DROPOUT		// not implemented
};


template<typename T = DEFAULT_SCALAR>
class NeuralNetwork {
	static_assert(std::is_arithmetic<T>::value, "Template parameter T for class NeuralNetwork must be arithmetic");
public:
	using Matrix	=	Matrix_<T>;
	using VecR		=	VecR_<T>;
	using VecC		=	VecC_<T>;
	using IOList	=	IOList_<T>;
	using DataSet	=	DataSet_<T>;

	typedef std::unique_ptr<VecR>		Layer_t;
	typedef std::unique_ptr<Matrix>		Weight_t;
	typedef std::vector<Layer_t>		Layers_t;
	typedef std::vector<Weight_t>		Weights_t;
	typedef std::function<T(T)>			UnaryFunc_t;


	NeuralNetwork(
		const std::vector<size_t>& topo,
		float learning_rate = 0.005,
		ActivationFunc func = ActivationFunc::SIGMOID,
		Regularization reg = Regularization::NONE,
		float reg_rate = 0.003
	);
	NeuralNetwork(
		std::vector<size_t>&& topo,
		float learning_rate = 0.005,
		ActivationFunc func = ActivationFunc::SIGMOID,
		Regularization reg = Regularization::NONE,
		float reg_rate = 0.003
	);
	NeuralNetwork(
		Weights_t&& weights,
		float learning_rate = 0.005,
		ActivationFunc func = ActivationFunc::SIGMOID,
		Regularization reg = Regularization::NONE,
		float reg_rate = 0.003
	);

	void regenerate();
	void regenerate(std::vector<size_t>&&);
	void regenerate(Weights_t&&);

	void remix();

	void propegateForward(const VecR& input) const;
	void propegateBackward(const VecR& output);
	void calcErrors(const VecR& output);
	void updateWeights();

	inline void train(const DataSet& d)
		{ this->train(d.first, d.second); }
	void train(const IOList& data_in, const IOList& data_out);
	inline void train_verbose(const DataSet& d)
		{ this->train_verbose(d.first, d.second); }
	void train_verbose(const IOList& data_in, const IOList& data_out);
	inline void train_graph(const DataSet& d, std::vector<float>& p)
		{ this->train_graph(d.first, d.second, p); }
	void train_graph(const IOList& data_in, const IOList& data_out, std::vector<float>& progress);
	inline void train_instance(const DataPoint& d)
		{ this->train_instance(*d.first, *d.second); }
	float train_instance(const VecR& in, const VecR& out);
	void inference(const VecR& in, VecR& out) const;

	void export_weights(std::ostream& out);
	static void parse_weights(std::istream& in, Weights_t& weights);

	void setActivationFunc(ActivationFunc);
	void setRegularization(Regularization);
	inline void setLearningRate(float lr) { this->learning_rate = lr; }
	inline void setRegularizationRate(float r) { this->reg_rate = r; }
	inline const float getLearningRate() const { return this->learning_rate; }
	inline const float getRegularizationRate() const { return this->reg_rate; }
	inline const size_t inputs() const { return this->topology.front(); }
	inline const size_t outputs() const { return this->topology.back(); }

	//size_t computeHorizontalUnits() const;

	template<typename _T>
	inline bool compatibleDataSet(const DataSet_<_T>& d) const {
		return std::is_same<T, _T>::value &&
			(d.first.size() > 0 && this->inputs() == d.first[0]->size()) &&
			(d.second.size() > 0 && this->outputs() == d.second[0]->size());
	}

	void dump(std::ostream&);


protected:
	std::vector<size_t> topology;
	float learning_rate, reg_rate;

	UnaryFunc_t
		activation_func,
		activation_func_deriv,
		regularization_func,
		regularization_func_deriv;
	Regularization reg_f;

	mutable Layers_t
		neurons_matx,
		cache_matx;
	Layers_t errors;
	Weights_t weights;


};





#define ASSERT_NUMERIC(T)	static_assert(std::is_arithmetic<(T)>::value, "Type must be arithmetic");

template<typename T> inline static T sigmoid(T x)			{ ASSERT_NUMERIC(T) return (T)1 / (1 + exp(-x)); }
template<typename T> inline static T hyperbolictan(T x)		{ ASSERT_NUMERIC(T) return tanh(x); }
template<typename T> inline static T relu(T x)				{ ASSERT_NUMERIC(T) return x > 0 ? x : 0; }
template<typename T> inline static T linear(T x)			{ ASSERT_NUMERIC(T) return x; }
template<typename T> inline static T sin(T x)				{ ASSERT_NUMERIC(T) return sin(x); }
template<typename T> inline static T cos(T x)				{ ASSERT_NUMERIC(T) return cos(x); }
template<typename T> inline static T sigmoid_d(T x)			{ ASSERT_NUMERIC(T) return sigmoid(x) * (1 - sigmoid(x)); }
template<typename T> inline static T hyperbolictan_d(T x)	{ ASSERT_NUMERIC(T) return 1 - (hyperbolictan(x) * hyperbolictan(x)); }
template<typename T> inline static T relu_d(T x)			{ ASSERT_NUMERIC(T) return x > 0 ? (T)1 : (T)0; }
template<typename T> inline static T linear_d(T x)			{ ASSERT_NUMERIC(T) return (T)1; }
template<typename T> inline static T sin_d(T x)				{ ASSERT_NUMERIC(T) return cos(x); }
template<typename T> inline static T cos_d(T x)				{ ASSERT_NUMERIC(T) return -sin(x); }

template<typename T> inline static T reg_L1(T w)			{ ASSERT_NUMERIC(T) return abs(w); }
template<typename T> inline static T reg_L2(T w)			{ ASSERT_NUMERIC(T) return 0.5 * w * w; }
template<typename T> inline static T reg_L1_d(T w)			{ ASSERT_NUMERIC(T) return w < 0 ? -1 : (w > 0 ? 1 : 0); }
template<typename T> inline static T reg_L2_d(T w)			{ ASSERT_NUMERIC(T) return w; }

template<typename T>
inline static T activation(T x, ActivationFunc f) {
	ASSERT_NUMERIC(T)
	switch (f) {
		case SIGMOID: return sigmoid<T>(x);
		case TANH: return hyperbolictan<T>(x);
		case RELU: return relu<T>(x);
		case LINEAR: return linear<T>(x);
		case SINE: return sin<T>(x);
		case COSINE: return cos<T>(x);
	}
}
template<typename T>
inline static std::function<T(T)> getFunc(ActivationFunc f) {
	ASSERT_NUMERIC(T)
	switch (f) {
		case SIGMOID: return sigmoid<T>;
		case TANH: return hyperbolictan<T>;
		case RELU: return relu<T>;
		case LINEAR: return linear<T>;
		case SINE: return sin<T>;
		case COSINE: return cos<T>;
	}
}
template<typename T>
inline static T activation_deriv(T x, ActivationFunc f) {
	ASSERT_NUMERIC(T)
	switch (f) {
		case SIGMOID: return sigmoid_d<T>(x);
		case TANH: return hyperbolictan_d<T>(x);
		case RELU: return relu_d<T>(x);
		case LINEAR: return 1;
		case SINE: return sin_d<T>(x);
		case COSINE: return cos_d<T>(x);
	}
}
template<typename T>
inline static std::function<T(T)> getFuncDeriv(ActivationFunc f) {
	ASSERT_NUMERIC(T)
	switch (f) {
		case SIGMOID: return sigmoid_d<T>;
		case TANH: return hyperbolictan_d<T>;
		case RELU: return relu_d<T>;
		case LINEAR: return linear<T>;
		case SINE: return sin_d<T>;
		case COSINE: return cos_d<T>;
	}
}
template<typename T>
inline static std::function<T(T)> getRegFunc(Regularization f) {
	ASSERT_NUMERIC(T)
	switch (f) {
		case NONE: return nullptr;
		case L1: return reg_L1<T>;
		case L2: return reg_L2<T>;
		case DROPOUT: return nullptr;
		default: return nullptr;
	}
}
template<typename T>
inline static std::function<T(T)> getRegFuncDeriv(Regularization f) {
	ASSERT_NUMERIC(T)
	switch (f) {
		case NONE: return nullptr;
		case L1: return reg_L1<T>;
		case L2: return reg_L2<T>;
		case DROPOUT: return nullptr;
		default: return nullptr;
	}
}



template<typename T> void exportData(DataSet_<T>& d, std::ostream& o) {
	for (size_t s = 0; s < d.first.size(); s++) {
		VecR_<T>& ins = *d.first[s];
		VecR_<T>& outs = *d.second[s];
		for (size_t i = 0; i < ins.size(); i++) {
			o << ins[i];
			if (i != ins.size() - 1) { o << ", "; }
		}
		o << " : ";
		for (size_t i = 0; i < outs.size(); i++) {
			o << outs[i];
			if (i != outs.size() - 1) { o << ", "; }
		}
		o << "\n";
	}
	o.flush();
}
template<typename T> void exportData_strictCSV(DataSet_<T>& d, std::ostream& oi, std::ostream& oo) {
	for (size_t s = 0; s < d.first.size(); s++) {
		VecR_<T>& ins = *d.first[s];
		VecR_<T>& outs = *d.second[s];
		for (size_t i = 0; i < ins.size(); i++) {
			oi << ins[i];
			oi << (i != ins.size() - 1 ? ", " : "\n");
		}
		oi.flush();
		for (size_t i = 0; i < outs.size(); i++) {
			oo << outs[i];
			oo << (i != outs.size() - 1 ? ", " : "\n");
		}
		oo.flush();
	}
}
template<typename T> void importData(DataSet_<T>& d, std::istream& i) {
	std::string sect;
	std::vector<T> buff;
	size_t pos, split;
	while (std::getline(i, sect, '\n')) {
		buff.clear();
		pos = split = 0;
		std::istringstream stream(sect);
		split = sect.find(':', 0);
		if (split != std::string::npos) {
			for (;;) {
				buff.emplace_back();
				stream >> buff.back();
				pos = sect.find(',', stream.tellg());
				if (stream.tellg() == (std::istringstream::pos_type)(-1)) {
					d.second.emplace_back(std::make_unique<VecR_<T> >(
						Eigen::Map<VecR_<T> >(buff.data(), buff.size())
					));
					break;
				} else if (pos - stream.tellg() < 3) {
					stream.ignore(2, ',');
				} else if (pos > split && (split - stream.tellg()) < 3) {
					d.first.emplace_back(std::make_unique<VecR_<T> >(
						Eigen::Map<VecR_<T> >(buff.data(), buff.size())
					));
					buff.clear();
					stream.ignore(2, ':');
				} else {
					// ???
				}
			}
		} else {
			// no separator
		}
		if (d.first.size() > 1 && d.first.back()->size() != d.first[d.first.size() - 2]->size()) {
			// error
		}
		if (d.second.size() > 1 && d.second.back()->size() != d.second[d.second.size() - 2]->size()) {
			// error
		}
		
	}
}
template<typename T> void importData_strictCSV(DataSet_<T>&, std::istream&, std::istream&) {}



template<typename T>
void NeuralNetwork<T>::regenerate() {
	this->neurons_matx.clear();
	this->cache_matx.clear();
	this->errors.clear();
	this->weights.clear();

	for (size_t i = 0; i < this->topology.size(); i++) {
		if (i == topology.size() - 1) {		// last(output) layer
			this->neurons_matx.emplace_back(	// add layer of size of last layer in topo
				std::make_unique<VecR>(this->topology.at(i))	);
		} else {
			this->neurons_matx.emplace_back(	// add layer of size of match topo + 1 for bias
				std::make_unique<VecR>(this->topology.at(i) + 1)	);
		}
		// topo{2, 3, 1} -> neurons{3, 4, 1}

		this->cache_matx.emplace_back(	// rvecs of the size of current neuron layer amount (?)
			std::make_unique<VecR>(this->neurons_matx.back()->size())	);	// changed from 'neurons_matx.size()'
		this->errors.emplace_back(			// ^^^
			std::make_unique<VecR>(this->neurons_matx.back()->size())	);	// '''
		this->cache_matx.back()->setZero();
		this->errors.back()->setZero();
		// topo{2, 3, 1} -> cache/delta{1, 2, 3} [x] -> cache/delta{3, 4, 1}

		if (i != this->topology.size() - 1) {	// not last idx
			this->neurons_matx.back()->coeffRef(this->topology.at(i)) = 1.0;	// last coeff(bias) in layer rvec = 1.0
			this->cache_matx.back()->coeffRef(this->topology.at(i)) = 1.0;	// accessing out of bounds?
		}
		/*
		* topo{2, 3, 1} ->
		* neurons
		* { {n, n, 1},
		*	{n, n, n, 1},
		*	{n} }
		* cache
		* { {0, 0, 1},
		*	{0, 0, 0, 1},
		*	{0} }
		* delta
		* { {0, 0, 0},
		*	{0, 0, 0, 0},
		*	{0} }
		*/

		if (i > 0) {	// not first idx
			if (i != topology.size() - 1) {	// and not last idx
				this->weights.emplace_back(
					std::make_unique<Matrix>(	// resizable matrix
						this->topology.at(i - 1) + 1,	// last layer size plus 1
						this->topology.at(i) + 1		// this layer size plus 1 for bias
					));
				this->weights.back()->setRandom();	// randomize starting weight
				this->weights.back()->col(this->topology.at(i)).setZero();	// set last col to all zeros
				this->weights.back()->coeffRef(	// set outermost coeff to 1 (highest of ^^^)
					this->topology.at(i - 1),
					this->topology.at(i)
				) = 1.0;
			} else {						// last idx
				this->weights.emplace_back(
					std::make_unique<Matrix>(
						this->topology.at(i - 1) + 1,	// last layer size plus 1
						this->topology.at(i)			// this layer size (no bias in output layer)
					));
				this->weights.back()->setRandom();	// assign random starting values
				this->weights.back()->row(this->weights.back()->rows() - 1).setZero();
			}
		}
		/*
		* topo{2, 3, 1} ->
		* weights(s)
		* { {3, 4},
		*	{4, 1} }
		* ~~~>>>
		* weights
		* { {
		*	{r, r, r, 0},
		*	{r, r, r, 0},
		*	{r, r, r, 1}
		* }, {
		*	{r},
		*	{r},
		*	{r},
		*	{r}
		* } }
		*/

	}
}
template<typename T>
void NeuralNetwork<T>::regenerate(std::vector<size_t>&& topo) {
	if (topo.size() > 1) {
		this->topology = std::move(topo);
		this->regenerate();
	}
}
template<typename T>
void NeuralNetwork<T>::regenerate(Weights_t&& weights) {
	if (!weights.empty()) {
		this->weights = std::move(weights);
		this->topology.clear();
		this->neurons_matx.clear();
		this->cache_matx.clear();
		this->errors.clear();

		this->topology.push_back( this->weights.at(0)->rows() - 1U );
		for (size_t i = 0; i < this->weights.size(); i++) {	// generate topo using the inverse of weight generation above
			this->topology.push_back(
				this->weights.at(i)->cols() - 1U );
		}
		this->topology.back()++;
		for (size_t i = 0; i < this->topology.size(); i++) {	// same as above
			if (i == topology.size() - 1) {
				this->neurons_matx.emplace_back(
					std::make_unique<VecR>(this->topology.at(i))	);
			} else {
				this->neurons_matx.emplace_back(
					std::make_unique<VecR>(this->topology.at(i) + 1)	);
			}

			this->cache_matx.emplace_back(
				std::make_unique<VecR>(this->neurons_matx.back()->size())	);
			this->errors.emplace_back(
				std::make_unique<VecR>(this->neurons_matx.back()->size())	);
			this->cache_matx.back()->setZero();
			this->errors.back()->setZero();

			if (i != this->topology.size() - 1) {
				this->neurons_matx.back()->coeffRef(this->topology.at(i)) = 1.0;
				this->cache_matx.back()->coeffRef(this->topology.at(i)) = 1.0;
			}

		}
	}
}

template<typename T>
void NeuralNetwork<T>::remix() {
	for (size_t i = 0; i < this->weights.size(); i++) {
		this->weights[i]->setRandom();
		if (i != this->weights.size() - 1) {
			this->weights[i]->col(this->weights[i]->cols() - 1).setZero();
			this->weights[i]->coeffRef(
				this->weights[i]->rows() - 1,
				this->weights[i]->cols() - 1
			) = 1.0;
		} else {
			this->weights.back()->row(this->weights.back()->rows() - 1).setZero();
		}
	}
}


template<typename T>
NeuralNetwork<T>::NeuralNetwork(
	const std::vector<size_t>& topo, float lr, ActivationFunc f, Regularization r, float rr
) : NeuralNetwork(std::move(std::vector<size_t>(topo)), lr, f, r, rr) {}
template<typename T>
NeuralNetwork<T>::NeuralNetwork(
	std::vector<size_t>&& topo, float lr, ActivationFunc f, Regularization r, float rr
) : 
	topology(std::move(topo)), learning_rate(lr), reg_rate(rr)
{
	this->setActivationFunc(f);
	this->setRegularization(r);
	this->regenerate();
}
template<typename T>
NeuralNetwork<T>::NeuralNetwork(
	Weights_t&& weights, float lr, ActivationFunc f, Regularization r, float rr
) :
	learning_rate(lr), reg_rate(rr)
{
	this->setActivationFunc(f);
	this->setRegularization(r);
	this->regenerate(std::move(weights));
}


template<typename T>
void NeuralNetwork<T>::propegateForward(const VecR& input) const {
	this->neurons_matx.front()->block(0, 0, 1, this->neurons_matx.front()->cols() - 1) = input;	// set the first neuron layer to input
	for (size_t i = 1; i < this->topology.size(); i++) {	// iterate through remaining layers
		if (i != this->topology.size() - 1) {
			*this->cache_matx[i] = (*this->neurons_matx[i - 1]) * (*this->weights[i - 1]);	// fill next neuron layer with previous * weights(connecting the 2 layers)
			this->neurons_matx[i]->block(0, 0, 1, this->neurons_matx[i]->cols() - 1) =
				this->cache_matx[i]->block(0, 0, 1, this->cache_matx[i]->cols() - 1).unaryExpr(this->activation_func);	// apply activation func to all neurons in layer
		} else {
			*this->neurons_matx[i] = (*this->neurons_matx[i - 1]) * (*this->weights[i - 1]);
		}
	}
}
template<typename T>
void NeuralNetwork<T>::propegateBackward(const VecR& output) {
	this->calcErrors(output);
	this->updateWeights();
}
template<typename T>
void NeuralNetwork<T>::calcErrors(const VecR& output) {
	*this->errors.back() =
		(output - *this->neurons_matx.back())/*.array() *
		this->cache_matx.back()->unaryExpr(this->activation_func_deriv).array()*/;
	for (size_t i = this->topology.size() - 2; i > 0; i--) {
		(*this->errors[i]) =
			((*this->errors[i + 1]) * (this->weights[i]->transpose())).array() *
			this->cache_matx[i]->unaryExpr(this->activation_func_deriv).array();	// store error for each layer by working backwards
		this->errors[i]->coeffRef(this->errors[i]->size() - 1) = 0;
		if (!this->errors[i]->allFinite()) {
			std::cout << "CalcErrors failure: NaN detected. Aborting." << std::endl;
			this->dump(std::cout);
			abort();
		}
	}
}
template<typename T>
void NeuralNetwork<T>::updateWeights() {
	size_t i = 0;
	for (; i < this->topology.size() - 1; i++) {
		if (this->reg_f == L2) {
			this->weights[i]->
				block(0, 0, this->weights[i]->rows() - 1, this->weights[i]->cols() - 1)
				*= (1.f - this->reg_rate * this->learning_rate);
		}
		else if (this->regularization_func_deriv) {
			this->weights[i]->
				block(0, 0, this->weights[i]->rows() - 1, this->weights[i]->cols() - 1)
				-= (this->weights[i]->
					block(0, 0, this->weights[i]->rows() - 1, this->weights[i]->cols() - 1).unaryExpr(
						this->regularization_func_deriv
					) * this->reg_rate * this->learning_rate);
		}
		*this->weights[i] += (
			this->learning_rate * (this->neurons_matx[i]->transpose() * *this->errors[i + 1])
		);
		if (!this->weights[i]->allFinite()) {
			std::cout << "UpdateWeights failure: NaN detected. Aborting." << std::endl;
			this->dump(std::cout);
			abort();
		}
	}
	this->weights[i - 1]->row(this->weights[i - 1]->rows() - 1).setZero();

////		if (i != this->topology.size() - 2) {
//			for (size_t c = 0; c < this->weights[i]->cols() - (i != this->topology.size() - 2); c++) {	// skip last col exept on last matx
//				for (size_t r = 0; r < this->weights[i]->rows() - 1; r++) {	// skip last row on last matx
//					this->weights[i]->coeffRef(r, c) +=
//						this->learning_rate *
//						this->errors[i + 1]->coeff(c) *
//						//this->activation_func_deriv(this->cache_matx[i + 1]->coeffRef(c)) *
//						this->neurons_matx[i]->coeff(r);	// for biases delete this part, just add lr * err
//						// also subtract regularization value
//				}
//				if (i != this->topology.size() - 2) {
//					this->weights[i]->coeffRef(this->weights[i]->rows() - 1, c) +=
//						this->learning_rate * this->errors[i + 1]->coeff(c);
//				}
//			}
//			
////		}
//		//else {
//		//	for (size_t c = 0; c < this->weights[i]->cols(); c++) {
//		//		for (size_t r = 0; r < this->weights[i]->rows(); r++) {
//		//			this->weights[i]->coeffRef(r, c) +=
//		//				this->learning_rate *
//		//				this->errors[i + 1]->coeff(c) *
//		//				//this->activation_func_deriv(this->cache_matx[i + 1]->coeffRef(c)) *
//		//				this->neurons_matx[i]->coeff(r);
//		//		}
//		//	}
//		//}
}
template<typename T>
void NeuralNetwork<T>::train(const IOList& data_in, const IOList& data_out) {
	for (size_t i = 0; i < data_in.size(); i++) {
		this->propegateForward(*data_in.at(i));
		this->propegateBackward(*data_out.at(i));
	}
}
template<typename T>
void NeuralNetwork<T>::train_verbose(const IOList& data_in, const IOList& data_out) {
	for (size_t i = 0; i < data_in.size(); i++) {
		std::cout << "Input to neural network is: " << *data_in.at(i) << '\n';
		this->propegateForward(*data_in.at(i));
		std::cout << "Expected output is: " << *data_out.at(i) <<
			"\nOutput produced is: " << *this->neurons_matx.back() << '\n';
		this->propegateBackward(*data_out.at(i));
		std::cout << "Itr: " << i << " - MSE: " << std::sqrt(
			(*this->errors.back()).dot(*this->errors.back()) / this->errors.back()->size()
		) << '\n' << std::endl;
	}
	/*std::cout << "Weights:\n";
	for (size_t i = 0; i < this->weights.size(); i++) {
		std::cout << "{\n" << *this->weights.at(i) << "\n}\n";
	}*/
}
template<typename T>
void NeuralNetwork<T>::train_graph(const IOList& data_in, const IOList& data_out, std::vector<float>& progress) {
	//progress.clear();
	for (size_t i = 0; i < data_in.size(); i++) {
		this->propegateForward(*data_in.at(i));
		this->propegateBackward(*data_out.at(i));
		progress.push_back(std::sqrt(
			(*this->errors.back()).dot(*this->errors.back()) / this->errors.back()->size()));
	}
}
template<typename T>
float NeuralNetwork<T>::train_instance(const VecR& in, const VecR& out) {
	this->propegateForward(in);
	propegateBackward(out);
	return std::sqrt(
		(*this->errors.back()).dot(*this->errors.back()) / this->errors.back()->size());
}
template<typename T>
void NeuralNetwork<T>::inference(const VecR& in, VecR& out) const {
	this->propegateForward(in);
	out = *this->neurons_matx.back();
}


template<typename T>
void NeuralNetwork<T>::export_weights(std::ostream& out) {
	size_t wm = this->weights.size(), rm, cm;
	for (size_t w = 0; w < wm; w++) {
		out << "{\n";
		rm = this->weights[w]->rows();
		cm = this->weights[w]->cols();
		for (size_t r = 0; r < rm; r++) {
			out << "\t[ ";
			for (size_t c = 0; c < cm; c++) {
				out << this->weights[w]->coeff(r, c) << ' ';	// or '\t'
			}
			out << (r == rm - 1 ? "]\n" : "],\n");
		}
		out << (w == wm - 1 ? "}\n" : "},\n");
	}
}
template<typename T>
void NeuralNetwork<T>::parse_weights(std::istream& in, Weights_t& weights) {
	std::string line;
	std::vector<std::vector<Scalar_t> > buff;
	//size_t r = 0, c = 0;
	weights.clear();
	while (std::getline(in, line, '\n')) {
		if (line.length() > 0 && line.at(0) == '{') {
			// start new weights block
			buff.clear();
			while (std::getline(in, line, '\n') && line.at(0) != '}') {
				buff.emplace_back();
				std::istringstream str(line);
				str.ignore(2, '[');
				while (str.ignore(1) && str.peek() != ']') {
					buff.back().emplace_back();
					str >> buff.back().back();
					//std::cout << buff.back().back() << std::endl;
				}
			}
			weights.emplace_back(
				std::make_unique<Matrix>(
					buff.size(),
					buff[0].size()
					)
			);
			for (size_t r = 0; r < buff.size(); r++) {
				for (size_t c = 0; c < buff[r].size(); c++) {
					weights.back()->coeffRef(r, c) = buff[r][c];
					//std::cout << buff[r][c] << std::endl;
				}
			}
			if (line.length() == 1 && line.at(0) == '}') {
				return;
			}
		}
	}
}
template<typename T>
void NeuralNetwork<T>::setActivationFunc(ActivationFunc f) {
	this->activation_func = getFunc<T>(f);
	this->activation_func_deriv = getFuncDeriv<T>(f);
}
template<typename T>
void NeuralNetwork<T>::setRegularization(Regularization f) {
	this->reg_f = f;
	this->regularization_func = getRegFunc<T>(f);
	this->regularization_func_deriv = getRegFuncDeriv<T>(f);
}


// template<typename T>
// size_t NeuralNetwork<T>::computeHorizontalUnits() const {
// 	size_t ret{ this->topology.size() };
// 	for (size_t i = 0; i < this->weights.size(); i++) {
// 		ret += this->weights[i]->cols();
// 	}
// 	return ret;
// }

template<typename T>
void NeuralNetwork<T>::dump(std::ostream& out) {
	out << "Weights:\n";
	for (size_t i = 0; i < this->weights.size(); i++) {
		out << '[' <<  *this->weights[i] << "]\n";
	}
	out.flush();
	out << "\nNeurons:\n";
	for (size_t i = 0; i < this->neurons_matx.size(); i++) {
		out << '[' << *this->neurons_matx[i] << "]\n";
	}
	out.flush();
	out << "\nCaches:\n";
	for (size_t i = 0; i < this->neurons_matx.size(); i++) {
		out <<  '[' << *this->cache_matx[i] << "]\n";
	}
	out.flush();
	out << "\nDeltas:\n";
	for (size_t i = 0; i < this->errors.size(); i++) {
		out << '[' << *this->errors[i] << "]\n";
	}
	out << "\n\n";
	out.flush();
}











//template<size_t... Args> struct select_first_size_t;
//template<size_t A, size_t ...Args> struct select_first_size_t<A, Args...> { static constexpr size_t value = A; };
//
//template<size_t A> struct tag{ static constexpr size_t value = A; };
//template<size_t... Args> struct select_last_size_t { static constexpr size_t value = decltype((tag<Args>{}, ...))::value; };
//
//
//template<size_t... topology>
//class NeuralNetwork_ {
//public:
//	static constexpr size_t
//		layers = sizeof...(topology),
//		inputs = select_first_size_t<topology...>::value,
//		outputs = select_last_size_t<topology...>::value;
//	static_assert(layers > 2, "Cannot construct NN with less than 2 layers");
//
//	typedef std::unique_ptr<VecR>				Layer_t;
//	typedef std::unique_ptr<Matrix>				Weight_t;
//	typedef std::array<Layer_t, layers>			Layers_t;
//	typedef std::array<Weight_t, layers - 1>	Weights_t;
//
//
//	inline NeuralNetwork_(Scalar_t lrate = 0.005) : learning_rate(lrate) {
//		this->init();
//	}
//	NeuralNetwork_(Weights_t&& weights);
//	NeuralNetwork_(MultiMat_Uq&& weights);
//
//	void propegateForward(const VecR& input) const;
//	void propegateBackward(VecR& output);
//	void calcErrors(VecR& output);
//	void updateWeights();
//	void train(MultiRow_Uq& data_in, MultiRow_Uq& data_out);
//	void train_verbose(MultiRow_Uq& data_in, MultiRow_Uq& data_out);
//	void train_graph(MultiRow_Uq& data_in, MultiRow_Uq& data_out, std::vector<float>& progress);
//	float train_instance(VecR& in, VecR& out);
//	void inference(const VecR& in, VecR& out) const;
//
//
//protected:
//	template<size_t l = 0, size_t... ls>
//	void init_(size_t i = 0, size_t last = 0) {
//		if constexpr (i == layers) {
//			return;
//		}
//		static_assert(l > 0, "Layer must have at least 1 neuron");
//
//		if constexpr (i == layers - 1) {
//			this->neurons_matx[i] = std::make_unique<VecR>(l);
//			this->cache_matx[i] = std::make_unique<VecR>(l);
//			this->errors[i] = std::make_unique<VecR>(l);
//		} else {
//			this->neurons_matx[i] = std::make_unique<VecR>(l + 1);
//			this->cache_matx[i] = std::make_unique<VecR>(l + 1);
//			this->errors[i] = std::make_unique<VecR>(l + 1);
//
//			this->neurons_matx[i]->coeffRef(l) = 1.0;
//			this->cache_matx[i]->coeffRef(l) = 1.0;
//		}
//
//		if constexpr (i > 0) {
//			if constexpr (i != layers - 1) {
//				this->weights[i] = std::make_unique<Matrix>(last + 1, l + 1);
//				this->weights[i]->setRandom();
//				this->weights[i]->col(l).setZero();
//				this->weights[i]->coeffRef(last + 1, l) = 1.0;
//			} else {
//				this->weights[i] = std::make_unique<Matrix>(last + 1, l);
//				this->weights[i]->setRandom();
//			}
//		}
//
//		init_<ls...>(i + 1, l);
//	}
//
//	inline void init() {
//		this->init_< topology... >();
//	}
//
//	mutable Layers_t neurons_matx;
//	Layers_t cache_matx, errors;
//	Weights_t weights;
//	Scalar_t learning_rate;
//
//
//};