#ifndef ADA_FASTFOOD_PARAM_H
#define ADA_FASTFOOD_PARAM_H

#include "iparam.h"
#include <vector>

template<MatMode mode, typename Dtype>
class DenseMat;

template<MatMode mode, typename Dtype>
class FastWHT;

template<MatMode mode, typename Dtype>
class AdaFastfoodParam : public IParam<mode, Dtype>
{
public:
		AdaFastfoodParam(FILE* fid);
		AdaFastfoodParam(std::string _name, GraphAtt _operand, size_t _input_size, size_t _output_size, BiasOption _bo = BiasOption::BIAS);
		AdaFastfoodParam(std::string _name, GraphAtt _operand, size_t _input_size, size_t _output_size, Dtype mean, Dtype std, BiasOption _bo = BiasOption::BIAS);
				
		void InitParams(Dtype mean, Dtype std);
		
		virtual void InitializeBatch(GraphData<mode, Dtype>* g) override;
		
		virtual void UpdateParams(Dtype lr, Dtype l2_penalty, Dtype momentum) override;
		
		virtual void Serialize(FILE* fid) override;
		virtual void Deserialize(FILE* fid) override;
		
		virtual void UpdateOutput(GraphData<mode, Dtype>* input_graph, DenseMat<mode, Dtype>* output, Dtype beta, Phase phase) override;
        virtual void UpdateGradInput(GraphData<mode, Dtype>* gradInput_graph, DenseMat<mode, Dtype>* gradOutput, Dtype beta) override;
        virtual void AccDeriv(GraphData<mode, Dtype>* input_graph, DenseMat<mode, Dtype>* gradOutput) override;
		
		virtual size_t OutSize() override
		{
			return output_size;
		}
		virtual size_t InSize() override
		{
			return input_size;
		}
		

		BiasOption bo;
		size_t input_size, output_size, pad_output_size, pad_input_size, num_parts;
		
		std::vector< DenseMat<mode, Dtype>* > binomial, gaussian, chisquare;
		std::vector< DenseMat<mode, Dtype>* > state_B, state_G, state_P, state_S;
		std::vector< DenseMat<mode, Dtype>* > grad_buf;
		
		DenseMat<mode, Dtype> state_input, grad_input, state_output, bias_multiplier, bias;
		std::vector< int* > perm, inv_perm;
		FastWHT<mode, Dtype>* wht_helper;
		
		bool grad_updated;
protected: 
		void InitStructures(); 
		void BackProp(DenseMat<mode, Dtype>* gradOutput);
};

#endif