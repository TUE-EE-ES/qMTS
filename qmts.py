import copy
import math
import logging as _logging

_logging_level = _logging.INFO
_logging_format = "%(asctime)s - %(name)s.%(funcName)s +%(lineno)s - %(levelname)s - %(message)s"
_logging.basicConfig(level=_logging_level, format=_logging_format)

logger = _logging.getLogger(__name__)

def PTQ_combostep(model, layer_index, acc_limit, depth = 2, PTQ_limit = 4):
    depth_val = 2**depth
    if(model.params[layer_index][1] <= PTQ_limit):
        return False
    if(model.params[layer_index][3] <= PTQ_limit):
        return False    

    orig_params = copy.deepcopy(model.params)
    model.params[layer_index][1] = int(model.params[layer_index][1]/depth_val)
    model.params[layer_index][0] = model.params[layer_index][0] * depth_val
    model.params[layer_index][3] = int(model.params[layer_index][3]/depth_val)
    model.params[layer_index][2] = model.params[layer_index][2] * depth_val
    best_acc = model.test()
    best_params = copy.deepcopy(model.params)
    print(best_acc)#, model.params)
    for i in range(depth):
        model.params[layer_index][0] = model.params[layer_index][0] / 2
        model.params[layer_index][2] = model.params[layer_index][2] / 2
        test_acc1 = model.test()
        print(test_acc1)#, model.params)
        if(test_acc1 > best_acc):
            best_acc = test_acc1
            best_params = copy.deepcopy(model.params)
    if(best_acc > acc_limit):
            model.params = copy.deepcopy(best_params)    
            print('Success: layer', layer_index, 'acc: ', best_acc, math.log2(model.params[layer_index][1])+1, 'bits, range' , model.params[layer_index][1]*model.params[layer_index][0])
            return True
    else:
            model.params = copy.deepcopy(orig_params)    
            print('Fail: layer', layer_index, 'acc: ', best_acc, math.log2(model.params[layer_index][1])+1, 'bits, range' , model.params[layer_index][1]*model.params[layer_index][0])
            return False

    # 1-weights 3-bias, index=layer, N bits, max value
def PTQ_step(model, layer_index, param_index, acc_limit, depth = 2, param_index_scale_offset = -1, PTQ_limit = 4):
    depth_val = 2**depth
    if(model.params[layer_index][param_index] <= PTQ_limit):
        return False
    if param_index == 1:
        param_str = "weight"
    else:
        param_str = "bias"
    orig_params = copy.deepcopy(model.params)
    model.params[layer_index][param_index] = int(model.params[layer_index][param_index]/depth_val)
    model.params[layer_index][param_index+param_index_scale_offset] = model.params[layer_index][param_index+param_index_scale_offset] * depth_val
    best_acc = model.test()
    best_params = copy.deepcopy(model.params)
    print(best_acc)#, model.params)
    for i in range(depth):
        model.params[layer_index][param_index+param_index_scale_offset] = model.params[layer_index][param_index+param_index_scale_offset] / 2
        test_acc1 = model.test()
        print(test_acc1)#, model.params)
        if(test_acc1 > best_acc):
            best_acc = test_acc1
            best_params = copy.deepcopy(model.params)
    if(best_acc > acc_limit):
            model.params = copy.deepcopy(best_params)    
            print('Success: layer', layer_index, param_str, 'acc: ', best_acc, math.log2(model.params[layer_index][param_index])+1, 'bits, range' , model.params[layer_index][param_index]*model.params[layer_index][param_index + param_index_scale_offset])
            return True
    else:
            model.params = copy.deepcopy(orig_params)    
            print('Fail: layer', layer_index, param_str, 'acc: ', best_acc, math.log2(model.params[layer_index][param_index])+1, 'bits, range' , model.params[layer_index][param_index]*model.params[layer_index][param_index + param_index_scale_offset])
            return False
    

def scale_calibrate(model, layer, param_index, acc_limit, param_index_scale_offset = -1, flag_offset = -2):

    model.dparams[layer][param_index+flag_offset] = True
    acc, _, _ = model.test()
    logger.info(f"accuracy: {acc}, params: {model.dparams}")
    while(acc <= acc_limit):
        model.dparams[layer][param_index] = int(2 * model.dparams[layer][param_index])
        model.dparams[layer][param_index+param_index_scale_offset] = model.dparams[layer][param_index+param_index_scale_offset]/2
        acc, _ , _= model.test()
        logger.info(f"accuracy: {acc}, params: {model.dparams}")


def range_constrain(model, layer, param_index, acc_limit, depth=4, param_index_scale_offset = -1, BIT_LIMIT = 4):
    acc, _, _ = model.test()
    logger.info(f"accuracy: {acc}, params: {model.dparams}")
    if param_index == 2:
        param_str = 'w'
        bits = int(1)
    else:
        param_str = 'b'
        bits = int(0)
    logger.info(f"Running Range constraint: layer: {layer} param: {param_str}")

    orig_params = copy.deepcopy(model.dparams)
    best_acc = 0
    for i in range(depth):
        model.dparams[layer][param_index] = int(model.dparams[layer][param_index]/2)
        if model.dparams[layer][param_index] < BIT_LIMIT:
            break
        acc, _, _ = model.test()
        if acc > best_acc:
            best_acc = acc
            best_params = copy.deepcopy(model.dparams)
        logger.info(f"accuracy: {acc}, params: {model.dparams}")
    if best_acc > acc_limit:
        model.dparams = copy.deepcopy(best_params)
        bits = math.log2(model.dparams[layer][param_index])+bits
        ranges = model.dparams[layer][param_index]*model.dparams[layer][param_index + param_index_scale_offset]         
        logger.info(f"Success: layer {layer}, {param_str} acc: {best_acc}, {bits} bits, range {ranges}")
        return True
    else:
        model.dparams = copy.deepcopy(orig_params)
        bits = math.log2(model.dparams[layer][param_index])+bits
        ranges = model.dparams[layer][param_index]*model.dparams[layer][param_index + param_index_scale_offset]         
        logger.info(f"Fail: layer {layer}, {param_str} acc: {best_acc}, {bits} bits, range {ranges}")
        return False        
    
    


def QAT_step(model, layer_index, param_index, acc_limit, qat_epochs = 25, depth = 2, param_index_scale_offset = -1, QAT_limit = 4):
    depth_val = 2**depth
    if(model.params[layer_index][param_index] <= QAT_limit):
       #flag[index] = False
       return False 
    if param_index == 1:
        param_str = "weight"
    else:
        param_str = "bias"
    orig_model = copy.deepcopy(model.state_dict())
    orig_params = copy.deepcopy(model.params)
    model.params[layer_index][param_index] = int(model.params[layer_index][param_index]/depth_val)
    model.params[layer_index][param_index+param_index_scale_offset] = model.params[layer_index][param_index+param_index_scale_offset] * depth_val
    print(model.params)
    best_acc, best_model = model.finetune(qat_epochs)
    best_params = copy.deepcopy(model.params)
    for i in range(depth):

        model.load_state_dict(copy.deepcopy(orig_model))
        model.params[layer_index][param_index+param_index_scale_offset] = model.params[layer_index][param_index+param_index_scale_offset] / 2
        print(model.params)
        acc, model_n = model.finetune(qat_epochs)
        if(acc > best_acc):
            best_model = model_n
            best_acc = acc
            best_params = copy.deepcopy(model.params)

    #w_scale[index] = w_scale[index] / depth_val # math.pow(2, depth - 1)
    if(best_acc > acc_limit):
        model.params = copy.deepcopy(best_params)
        model.load_state_dict(copy.deepcopy(best_model))
        print('Success: layer', layer_index, param_str, 'acc: ', best_acc, math.log2(model.params[layer_index][param_index])+1, 'bits, range' , model.params[layer_index][param_index]*model.params[layer_index][param_index + param_index_scale_offset])
        return True
    else:
        model.params = copy.deepcopy(orig_params)
        model.load_state_dict(copy.deepcopy(orig_model))
        #print(test(model))
        print('Fail: layer', layer_index, param_str, 'acc: ', best_acc, math.log2(model.params[layer_index][param_index])+1, 'bits, range' , model.params[layer_index][param_index]*model.params[layer_index][param_index + param_index_scale_offset])
        return False
    
def QAT_combostep(model, layer_index,acc_limit, qat_epochs = 25, depth = 2,  QAT_limit = 4):
    depth_val = 2**depth
    if(model.params[layer_index][1] <= QAT_limit):
       #flag[index] = False
       return False 
    if(model.params[layer_index][3] <= QAT_limit):
       #flag[index] = False
       return False 
    orig_model = copy.deepcopy(model.state_dict())
    orig_params = copy.deepcopy(model.params)
    model.params[layer_index][1] = int(model.params[layer_index][1]/depth_val)
    model.params[layer_index][0] = model.params[layer_index][0] * depth_val
    model.params[layer_index][3] = int(model.params[layer_index][3]/depth_val)
    model.params[layer_index][2] = model.params[layer_index][2] * depth_val    
    print(model.params)
    best_acc, best_model = model.finetune(qat_epochs)
    best_params = copy.deepcopy(model.params)
    for i in range(depth):

        model.load_state_dict(copy.deepcopy(orig_model))
        model.params[layer_index][0] = model.params[layer_index][0] / 2
        model.params[layer_index][2] = model.params[layer_index][2] / 2
        print(model.params)
        acc, model_n = model.finetune(qat_epochs)
        if(acc > best_acc):
            best_model = model_n
            best_acc = acc
            best_params = copy.deepcopy(model.params)

    #w_scale[index] = w_scale[index] / depth_val # math.pow(2, depth - 1)
    if(best_acc > acc_limit):
        model.params = copy.deepcopy(best_params)
        model.load_state_dict(copy.deepcopy(best_model))
        print('Success: layer', layer_index, 'acc: ', best_acc, math.log2(model.params[layer_index][1])+1, 'bits, range' , model.params[layer_index][1]*model.params[layer_index][0])
        return True
    else:
        model.params = copy.deepcopy(orig_params)
        model.load_state_dict(copy.deepcopy(orig_model))
        #print(test(model))
        print('Fail: layer', layer_index, 'acc: ', best_acc, math.log2(model.params[layer_index][1])+1, 'bits, range' , model.params[layer_index][1]*model.params[layer_index][0])
        return False    
    


def T_step(model, layer_index, acc_limit, has_bias = True, qat_epochs = 25):
    if has_bias:
        if(model.params[layer_index][1] != 2 or model.params[layer_index][3] != 2):
            return
    else:
        if(model.params[layer_index][1] != 2):
            return 
    #orig_model = copy.deepcopy(model.state_dict())
    orig_params = copy.deepcopy(model.params)
    model.params[layer_index][1] = 1
    if has_bias:
        model.params[layer_index][3] = 1
    logger.info(model.params)
    #best_acc = model.test()
    #best_acc, best_model = model.finetune(qat_epochs)
    best_params = copy.deepcopy(model.params)
    logger.info(best_acc)
    #model.load_state_dict(copy.deepcopy(orig_model))

    model.params[layer_index][0] = model.params[layer_index][0] * 2
    if has_bias:
        model.params[layer_index][2] = model.params[layer_index][2] * 2
    logger.info(model.params)
    #acc, model_n = model.finetune(qat_epochs)
    acc = model.test()
    logger.info(acc)
    if(acc > best_acc):
        #best_model = model_n
        best_acc = acc
        best_params = copy.deepcopy(model.params)

    #w_scale[index] = w_scale[index] / depth_val # math.pow(2, depth - 1)
    if(best_acc > acc_limit):
        model.params = copy.deepcopy(best_params)
        #model.load_state_dict(copy.deepcopy(best_model))
        bits = math.log2(model.params[layer_index][1])+1
        ranges = model.params[layer_index][1]*model.params[layer_index][0]
        logger.info(f"Success: layer{layer_index}, acc: {best_acc}, {bits} bits, range: {ranges}")
        return True
    else:
        model.params = copy.deepcopy(orig_params)
        #model.load_state_dict(copy.deepcopy(orig_model))
        #logger.info(test(model))
        bits = math.log2(model.params[layer_index][1])+1
        ranges = model.params[layer_index][1]*model.params[layer_index][0]
        logger.info(f"Fail: layer{layer_index}, acc: {best_acc}, {bits} bits, range: {ranges}")
        return False    