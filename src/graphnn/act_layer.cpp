#include "relu_layer.h"
#include "sin_layer.h"
#include "cos_layer.h"
#include "exp_layer.h"
#include "softmax_layer.h"
#include "dense_matrix.h"
#include "mkl_helper.h"

// =========================================== relu layer ================================================
template<typename Dtype>
void ReLULayer<CPU, Dtype>::Act(DenseMat<CPU, Dtype>& prev_out, DenseMat<CPU, Dtype>& cur_out)
{
        for (size_t i = 0; i < cur_out.count; ++i)
            cur_out.data[i] = prev_out.data[i] > 0 ? prev_out.data[i] : 0; 
}

template<typename Dtype>
void ReLULayer<CPU, Dtype>::Derivative(DenseMat<CPU, Dtype>& dst, DenseMat<CPU, Dtype>& prev_output, 
                            DenseMat<CPU, Dtype>& cur_output, DenseMat<CPU, Dtype>& cur_grad)
{
        dst.CopyFrom(cur_grad);
        for (int i = 0; i < dst.count; ++i)
            if (cur_output.data[i] <= 0.0)
                dst.data[i] = 0.0;
}

template class ReLULayer<CPU, float>;
template class ReLULayer<CPU, double>;

// =========================================== sin layer ================================================

template class SinLayer<CPU, float>;
template class SinLayer<CPU, double>;

// =========================================== cos layer ================================================

template class CosLayer<CPU, float>;
template class CosLayer<CPU, double>;

// =========================================== exp layer ================================================

template class ExpLayer<CPU, float>;
template class ExpLayer<CPU, double>;

// =========================================== softmax layer ================================================

template class SoftmaxLayer<CPU, float>;
template class SoftmaxLayer<CPU, double>;

/*
// =========================================== softmax layer ================================================
template<MatMode mode, typename Dtype>
SoftmaxLayer<mode, Dtype>::SoftmaxLayer(std::string _name, WriteType _wt, ActTarget _at, PropErr _properr)
							 : IActLayer<mode, Dtype>(_name, _wt, _at, _properr)
{
		this->graph_output = new GraphData<mode, Dtype>(DENSE);		
		this->graph_gradoutput = new GraphData<mode, Dtype>(DENSE);
}

template<typename Dtype>
void SoftmaxAct(DenseMat<CPU, Dtype>& mat)
{
		//#pragma omp parallel for
		Dtype sum, max_v;
		size_t i;
		Dtype* data;
		for (i = 0, data = mat.data; i < mat.rows; ++i, data += mat.cols)
		{
			max_v = data[0];
			for (size_t j = 1; j < mat.cols; ++j)
				max_v = max_v > data[j] ? max_v : data[j];
			for (size_t j = 0; j < mat.cols; ++j)
				data[j] -= max_v;						
		}
		
		MKLHelper_Exp(mat.count, mat.data, mat.data);
		
		data = mat.data;		
		for (i = 0, data = mat.data; i < mat.rows; ++i, data += mat.cols)
		{
			sum = MKLHelper_Asum(mat.cols, data);
			for (size_t j = 0; j < mat.cols; ++j)
				data[j] /= sum;
		}											
}

template<typename Dtype>
void SoftmaxAct(DenseMat<GPU, Dtype>& mat)
{
		throw "not implemented";
}

template<MatMode mode, typename Dtype>
void SoftmaxLayer<mode, Dtype>::UpdateOutput(ILayer<mode, Dtype>* prev_layer, SvType sv, Phase phase)
{		
		assert(sv == SvType::WRITE2);
		
		auto* prev_output = prev_layer->graph_output;				
		this->graph_output->graph = prev_output->graph;
		
		if (this->wt == WriteType::INPLACE)
		{
			this->graph_output->node_states = prev_output->node_states;
			this->graph_output->edge_states = prev_output->edge_states; 
		} else 
		{
			this->graph_output->node_states->DenseDerived().CopyFrom(prev_output->node_states->DenseDerived());
			this->graph_output->edge_states->DenseDerived().CopyFrom(prev_output->edge_states->DenseDerived());
		}
		
		if (this->at == ActTarget::NODE || this->at == ActTarget::NODE_EDGE)
			SoftmaxAct(this->graph_output->node_states->DenseDerived());
		if (this->at == ActTarget::EDGE || this->at == ActTarget::NODE_EDGE)
			SoftmaxAct(this->graph_output->edge_states->DenseDerived());
}

template<MatMode mode, typename Dtype>
void SoftmaxLayer<mode, Dtype>::BackPropErr(ILayer<mode, Dtype>* prev_layer, SvType sv)
{
		assert(sv == SvType::WRITE2);
		
		if (this->at == ActTarget::NODE || this->at == ActTarget::NODE_EDGE)
		{
			auto& grad_prev = prev_layer->graph_gradoutput->node_states->DenseDerived();
			
			Dtype* grad_data = this->graph_gradoutput->node_states->DenseDerived().data;
			Dtype* act_data = this->graph_output->node_states->DenseDerived().data;
									
			grad_prev.Resize(this->graph_output->node_states->rows, this->graph_output->node_states->cols);
			
			for (int i = 0; i < grad_prev.count; ++i)
				grad_prev.data[i] = grad_data[i] * act_data[i] * (1 - act_data[i]);				
		}
		
		if (this->at == ActTarget::EDGE || this->at == ActTarget::NODE_EDGE)
		{
			throw "not implemented";
		}
}

template class SoftmaxLayer<CPU, float>;
template class SoftmaxLayer<CPU, double>;

// =========================================== sigmoid layer ================================================

template<MatMode mode, typename Dtype>
SigmoidLayer<mode, Dtype>::SigmoidLayer(std::string _name, WriteType _wt, ActTarget _at, PropErr _properr)
							 : IActLayer<mode, Dtype>(_name, _wt, _at, _properr)
{
		this->graph_output = new GraphData<mode, Dtype>(DENSE);		
		this->graph_gradoutput = new GraphData<mode, Dtype>(DENSE);
}

template<typename Dtype>
void SigmoidAct(DenseMat<CPU, Dtype>& mat)
{
		for (size_t i = 0; i < mat.count; ++i)
			mat.data[i] = 1.0 / (1.0 + exp(-mat.data[i]));
}

template<typename Dtype>
void SigmoidAct(DenseMat<GPU, Dtype>& mat)
{
		throw "not implemented";
}

template<MatMode mode, typename Dtype>
void SigmoidLayer<mode, Dtype>::UpdateOutput(ILayer<mode, Dtype>* prev_layer, SvType sv, Phase phase)
{		
		assert(sv == SvType::WRITE2);
		
		auto* prev_output = prev_layer->graph_output;
		this->graph_output->graph = prev_output->graph;
		
		if (this->wt == WriteType::INPLACE)
		{
			this->graph_output->node_states = prev_output->node_states;
			this->graph_output->edge_states = prev_output->edge_states; 
		} else 
		{
			this->graph_output->node_states->DenseDerived().CopyFrom(prev_output->node_states->DenseDerived());
			this->graph_output->edge_states->DenseDerived().CopyFrom(prev_output->edge_states->DenseDerived());
		}
		
		if (this->at == ActTarget::NODE || this->at == ActTarget::NODE_EDGE)
			SigmoidAct(this->graph_output->node_states->DenseDerived());
		if (this->at == ActTarget::EDGE || this->at == ActTarget::NODE_EDGE)
			SigmoidAct(this->graph_output->edge_states->DenseDerived());
}

template<MatMode mode, typename Dtype>
void SigmoidLayer<mode, Dtype>::BackPropErr(ILayer<mode, Dtype>* prev_layer, SvType sv)
{
		if (this->at == ActTarget::NODE || this->at == ActTarget::NODE_EDGE)
		{
			auto& grad_prev = prev_layer->graph_gradoutput->node_states->DenseDerived();
			
			Dtype* grad_data = this->graph_gradoutput->node_states->DenseDerived().data;
			Dtype* act_data = this->graph_output->node_states->DenseDerived().data;
									
			grad_prev.Resize(this->graph_output->node_states->rows, this->graph_output->node_states->cols);
			
			for (int i = 0; i < grad_prev.count; ++i)
				grad_prev.data[i] = grad_data[i] * act_data[i] * (1 - act_data[i]);				 								
		}
		
		if (this->at == ActTarget::EDGE || this->at == ActTarget::NODE_EDGE)
		{
			throw "not implemented";
		}
}

template class SigmoidLayer<CPU, float>;
template class SigmoidLayer<CPU, double>;
*/