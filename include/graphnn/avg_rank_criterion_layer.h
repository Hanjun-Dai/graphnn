#ifndef AVG_RANK_CRITERION_LAYER_H
#define AVG_RANK_CRITERION_LAYER_H

#include "i_criterion_layer.h"

template<MatMode mode, typename Dtype>
class AvgRankCriterionLayer : public ICriterionLayer<mode, Dtype>
{
public:
			AvgRankCriterionLayer(std::string _name, RankOrder _order)
                : ICriterionLayer<mode, Dtype>(_name, PropErr::N), order(_order) {}
                        
            static std::string str_type()
            {
                return "AverageRank"; 
            }
            
            virtual void UpdateOutput(std::vector< ILayer<mode, Dtype>* >& operands, Phase phase) override
            {
                auto& pred = operands[0]->state->DenseDerived();
                auto& labels = operands[1]->state->SparseDerived();
                
                this->loss = LossFunc<mode, Dtype>::GetAverageRank(pred, labels, order);
            }
            
            virtual void BackPropErr(std::vector< ILayer<mode, Dtype>* >& operands, unsigned cur_idx, Dtype beta) override
            {
                throw std::runtime_error("no grad in this layer");
            }      
            
            RankOrder order;            
};

#endif