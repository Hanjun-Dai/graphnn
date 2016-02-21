#ifndef NNGRAPH_EXPR_H
#define NNGRAPH_EXPR_H

#include "i_comp_node.h"

template<MatMode mode, typename Dtype>
class NNGraphExpr 
{
public:    
    explicit NNGraphExpr(std::shared_ptr< ICompNode<mode, Dtype> > _comp_node)
        : comp_node(_comp_node)
        {
            args.clear();
        }
    
    explicit NNGraphExpr(std::shared_ptr< ICompNode<mode, Dtype> > _comp_node, std::vector<NNGraphExpr> _args) 
        : comp_node(_comp_node), args(_args)
        {          
        }
    
    inline NNGraphExpr<mode, Dtype>& operator=(const NNGraphExpr<mode, Dtype>& r_value)
    {
        args.clear();
        for (size_t i = 0; i < r_value.args.size(); ++i)
            args.push_back(r_value.args[i]);
        return *this;
    }
    
    std::shared_ptr< ICompNode<mode, Dtype> > comp_node;
    std::vector< NNGraphExpr<mode, Dtype> > args;    
};

template<MatMode mode, typename Dtype>
NNGraphExpr<mode, Dtype> operator+(const NNGraphExpr<mode, Dtype>& x, const NNGraphExpr<mode, Dtype>& y) 
{
    return NNGraphExpr<mode, Dtype>(std::make_shared< SumLayer<mode, Dtype> >(), {x, y});
}



#endif