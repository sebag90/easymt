import os
import argparse
import json
import re
from shutil import copyfile


def create_directories():
    os.mkdir ("../0-dataset")
    os.mkdir ("../1-norm")
    os.mkdir ("../2-tok") 
    os.mkdir ("../3-true")
    os.mkdir ("../3-true_models")
    os.mkdir ("../4-byte_data")
    os.mkdir ("../5-trained_models")
    os.mkdir ("../6-raw_pred")
    os.mkdir ("../6-detok_pred")


def cut_single_file(name, param, start, stop, language):
    os.system('sed -n "'+ str(start) +','+ str(stop)  +' p" ../../texts/' + param["dataset"][name] + '.'+ str(language) +
    '  > ../0-dataset/' + str(name) + '.' + str(language) + str(param["trained"]))


def copy_file(param, language, dataset):
    copyfile( "../../texts/" + str(param["dataset"][dataset]) + "." + str(language), 
    "../0-dataset/" + dataset +"." + str(language) + str(param["trained"]))


def normalize(name, param, language):
    os.system("perl ../preprocessing-tools/normalize-punctuation.perl -l " + str(language) +
    " < ../0-dataset/" + str(name) + "." + str(language) + str(param["trained"]) +
    " > ../1-norm/" + str(name) + ".norm." + str(language) + str(param["trained"]))


def tokenize(name, param, language):
    os.system("perl ../preprocessing-tools/tokenizer.perl -l " + str(language) + 
    " -no-escape < ../1-norm/" + str(name) + ".norm." + str(language)+ str(param["trained"]) + 
    " > ../2-tok/" + str(name) + ".tok." + str(language) + str(param["trained"]))


def train_truecase(language, param):
    os.system("perl ../preprocessing-tools/train-truecaser.perl -corpus ../2-tok/train.tok." + str(language) + str(param["trained"]) + 
    " -model ../3-true_models/truecasing." + str(language) + ".model" + str(param["trained"]))


def true_case(name, param, language):
    os.system("perl ../preprocessing-tools/truecase.perl -model ../3-true_models/truecasing." + str(language) + ".model"+ str(param["trained"]) + 
    " <  ../2-tok/" + str(name) + ".tok." + str(language) + str(param["trained"]) + 
    " > ../3-true/" + str(name) + ".true." + str(language) + str(param["trained"]))


def learn_bpe(param, language):
    os.system("subword-nmt learn-bpe -s "+ str(param["bpe"]) +" < ../3-true/train.true." + str(language) + str(param["trained"]) + 
    " > ../3-true_models/bpe."+ str(language) +".codes" + str(param["trained"] ))


def bpe_splitting(name, param, language):
    os.system("subword-nmt apply-bpe -c ../3-true_models/bpe." + str(language) + ".codes"+ str(param["trained"]) + 
    " < ../3-true/" + str(name) + ".true." + str(language) + str(param["trained"]) + 
    " > ../3-true/" + str(name) + ".bpe." + str(language) + str(param["trained"]))


def convert_byte(param):
    if param["bpe"] == 0:
        name = "true"
    else:
        name = "bpe"

    command = ("onmt_preprocess -train_src ../3-true/train." + name + "." + str(param["src_lang"]) + str(param["trained"]) + " " + 
                                "-train_tgt ../3-true/train." + name + "." + str(param["tgt_lang"]) + str(param["trained"]) + " " + 
                                "-valid_src ../3-true/val." + name + "."   + str(param["src_lang"]) + str(param["trained"]) + " " +  
                                "-valid_tgt ../3-true/val." + name + "."   + str(param["tgt_lang"]) + str(param["trained"]) + " " +  
                                "-save_data ../4-byte_data/data-" + str(param["src_lang"]) + "-" + str(param["tgt_lang"]) + str(param["trained"]))

    os.system(command)


def train(param):
    command = ("onmt_train -data ../4-byte_data/data-" + str(param["src_lang"]) + "-" + str(param["tgt_lang"]) + str(param["trained"]) + " "+
                            "-save_model ../5-trained_models/model-" + str(param["src_lang"]) + "-" + str(param["tgt_lang"]) + str(param["trained"]) + " "+
                            "-train_steps "            + str(param["total_steps"]*param["trained"]) + " "+
                            "-save_checkpoint_steps "  + str(param["save_valid"]) + " "+
                            "-valid_steps "            + str(param["save_valid"]) + " "+
                            "-global_attention "       + str(param["attention"]) + " "+
                            "-input_feed 0 " +
                            "-dropout 0.1 " +
                            "-world_size 1 " +
                            "-gpu_ranks 0 " +
                            "-layers "                 + str(param["layers"]) + " "+
                            "-rnn_size "               + str(param["rnn_size"]) + " "+
                            "-word_vec_size "          + str(param["word_vec_size"]) )

    if param["transformer"] == "yes":
        command += (" -transformer_ff 2048 -heads 8 " +
                    "-encoder_type transformer -decoder_type transformer -position_encoding "+
                    "-max_generator_batches 2 "+
                    "-batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2 "+
                    "-optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 "+
                    "-max_grad_norm 0 -param_init 0  -param_init_glorot " +
                    "-label_smoothing 0.1")
    
    if param["trained"] != 1:
        command += (" -train_from ../5-trained_models/model-" + str(param["src_lang"]) + "-" + str(param["tgt_lang"]) + str(param["trained"] - 1) + 
        "_step_" + str((param["trained"]-1)*100000) + ".pt")

        
    os.system(command)


def translate(param):
    if param["bpe"] == 0:
        source = "../3-true/test.true." + str(param["src_lang"]) + "1"
    else:
        source = "../3-true/test.bpe." + str(param["src_lang"]) + "1"


    command = ("onmt_translate -model ../5-trained_models/model-" + str(param["src_lang"]) + "-" + str(param["tgt_lang"]) + "*00000.pt " +  
                                "-src " + source  + " " +
                                "-output ../6-raw_pred/pred_" + str(param["tgt_lang"]) + str(param["trained"]) + " "  +
                                "-gpu 0")
    os.system(command)


def detokenize(param):
    if param["bpe"] == 0:
        os.system("perl ../preprocessing-tools/detokenizer.perl -u -l " + str(param["tgt_lang"]) + " < ../6-raw_pred/pred_" + str(param["tgt_lang"]) + 
        str(param["trained"]) + " > ../6-detok_pred/pred.detok." + str(param["tgt_lang"]) + str(param["trained"]))
    else:
        os.system("sed -r 's/(@@ )|(@@ ?$)//g' < ../6-raw_pred/pred_" + str(param["tgt_lang"]) + str(param["trained"]) + 
        " | perl ../preprocessing-tools/detokenizer.perl -u -l " + str(param["tgt_lang"]) + " > ../6-detok_pred/pred.detok." + 
        str(param["tgt_lang"]) + str(param["trained"]))


def bleu(param):
    bleu_score = os.popen("perl ../preprocessing-tools/multi-bleu-detok.perl ../0-dataset/test." + str(param["tgt_lang"])+ "1" + 
    " < ../6-detok_pred/pred.detok." + str(param["tgt_lang"]) + str(param["trained"])).read()
    clean_bleu = re.findall("^(.*?),", bleu_score)[0]

    param["last_bleu"] = param["dataset"]["test"] + ", " + str(clean_bleu)
    
    update_json(param, clean_bleu)


def update_json(param, bleu = None):
    # add used dataset
    used = {"Train":  param["dataset"]["train"] + " " +  str(param["dataset"]["begin_train"])  + "-" + str(param["dataset"]["begin_train"] + param["dataset"]["train_lines"]),
            "Val":    param["dataset"]["val"]   + " " +  str(param["dataset"]["begin_val"])  + "-" + str(param["dataset"]["val_lines"] + param["dataset"]["begin_val"]),
            "Test":   param["dataset"]["test"]
            }

    if param["dataset"]["begin_test"] == param["dataset"]["test_lines"]:
        used["Test"] += " complete"
    else:
        used["Test"] += " " + str(param["dataset"]["begin_test"])  + "-" + str(param["dataset"]["test_lines"] + param["dataset"]["begin_test"])

    if bleu != None:
        used["BLEU"] =  bleu.split()[2]

    param["models"][str(param["trained"])] = used
    # write to file
    with open("parameters.json", "w") as file:
        json.dump(param, file, indent=4)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-complete", help="preprocess, train and translate", action = "store_true")
    parser.add_argument("-translate", help="translate", action = "store_true")
    parser.add_argument("-new_test", help="preprocess a new test data set", action = "store_true")

    args = parser.parse_args()

    with open('parameters.json', 'r') as f:
        param = json.load(f)
    
    
    if args.tutto == True:

        if param["trained"] == 0:
            create_directories()
        param["trained"] += 1


        # cut files
        # TRAIN
        if param["dataset"]["begin_train"] == param["dataset"]["train_lines"]:
            copy_file(param, param["src_lang"], "train")
            copy_file(param, param["tgt_lang"], "train")
        else:
            cut_single_file("train", param, param["dataset"]["begin_train"], param["dataset"]["begin_train"] + param["dataset"]["train_lines"], param["src_lang"])
            cut_single_file("train", param, param["dataset"]["begin_train"], param["dataset"]["begin_train"] + param["dataset"]["train_lines"], param["tgt_lang"])


        #VAL
        if param["dataset"]["begin_val"] == param["dataset"]["val_lines"]:
            copy_file(param, param["src_lang"], "val")
            copy_file(param, param["tgt_lang"], "val")
        else:
            cut_single_file("val", param, param["dataset"]["begin_val"], param["dataset"]["begin_val"] + param["dataset"]["val_lines"], param["src_lang"])
            cut_single_file("val", param, param["dataset"]["begin_val"], param["dataset"]["begin_val"] + param["dataset"]["val_lines"], param["tgt_lang"])


        #TEST
        if param["trained"] == 1:
            if param["dataset"]["begin_test"] == param["dataset"]["test_lines"]:
                copy_file(param, param["src_lang"], "test")
                copy_file(param, param["tgt_lang"], "test")
            else:
                cut_single_file("test", param, param["dataset"]["begin_test"], param["dataset"]["begin_test"] + param["dataset"]["test_lines"], param["src_lang"])
                cut_single_file("test", param, param["dataset"]["begin_test"], param["dataset"]["begin_test"] + param["dataset"]["test_lines"], param["tgt_lang"])


        normalize
        normalize("train", param, param["src_lang"])
        normalize("train", param, param["tgt_lang"])
        normalize("val", param, param["src_lang"])
        normalize("val", param, param["tgt_lang"])

        if param["trained"] == 1:
            normalize("test", param, param["src_lang"])
           

        # tokenize
        tokenize("train", param, param["src_lang"])
        tokenize("train", param, param["tgt_lang"])
        tokenize("val", param, param["src_lang"])
        tokenize("val", param, param["tgt_lang"])

        if param["trained"] == 1:
            tokenize("test", param, param["src_lang"])
            

        # train truecase
        train_truecase(param["src_lang"], param)
        train_truecase(param["tgt_lang"], param)


        # truecasing
        true_case("train", param, param["tgt_lang"])
        true_case("train", param, param["src_lang"])
        true_case("val", param, param["tgt_lang"])
        true_case("val", param, param["src_lang"])

        if param["trained"] == 1:
            true_case("test", param, param["src_lang"])
         

        # bpe splitting
        if param["bpe"] != 0:
            learn_bpe(param, param["src_lang"])
            learn_bpe(param, param["tgt_lang"])

            bpe_splitting("train", param, param["src_lang"])
            bpe_splitting("train", param, param["tgt_lang"])
            bpe_splitting("val", param, param["src_lang"])
            bpe_splitting("val", param, param["tgt_lang"])

            if param["trained"] == 1:
                bpe_splitting("test", param, param["src_lang"])
             
                
        convert_byte(param)
        train(param)
        update_json(param)
        translate(param)
        detokenize(param)      
        bleu(param)
     

    elif args.translate == True:

        translate(param)
        detokenize(param)
        bleu(param)


    elif args.new_test == True:
        
        param["trained"] = 1
        copy_file(param, param["src_lang"], "test")
        copy_file(param, param["tgt_lang"], "test")
        normalize("test", param, param["src_lang"])
        tokenize("test", param, param["src_lang"])
        true_case("test", param, param["src_lang"])

        if param["bpe"] != 0:
            bpe_splitting("test", param, param["src_lang"])


if __name__ == "__main__":
    main()