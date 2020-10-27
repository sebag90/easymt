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
    command = (f'sed -n "{start},{stop}p" ../../texts/{param["dataset"][name]}.{language} > ../0-dataset/{name}.{language}{param["trained"]}')

    os.system(command)


def copy_file(param, language, dataset):
    copyfile(f"../../texts/{param['dataset'][dataset]}.{language}", f"../0-dataset/{dataset}.{language}{param['trained']}")


def normalize(name, param, language):
    command = (f"perl ../preprocessing-tools/normalize-punctuation.perl -l {language} < ../0-dataset/{name}.{language}{param['trained']}" +
               f" > ../1-norm/{name}.norm.{language}{param['trained']}")

    os.system(command)


def tokenize(name, param, language):
    command = (f"perl ../preprocessing-tools/tokenizer.perl -l {language} -no-escape < ../1-norm/{name}.norm.{language}{param['trained']}" + 
               f" > ../2-tok/{name}.tok.{language}{param['trained']}")

    os.system(command)

def train_truecase(language, param):
    command = (f"perl ../preprocessing-tools/train-truecaser.perl -corpus ../2-tok/train.tok.{language}{param['trained']}" + 
               f" -model ../3-true_models/truecasing.{language}.model({param['trained']}")

    os.system(command)

def true_case(name, param, language):
    command = (f"perl ../preprocessing-tools/truecase.perl -model ../3-true_models/truecasing.{language}.model{param['trained']}" + 
               f" <  ../2-tok/{name}.tok.{language}{param['trained']} > ../3-true/{name}.true.{language}{param['trained']}")

    os.system(command)


def learn_bpe(param, language):
    command = (f"subword-nmt learn-bpe -s {param['bpe']} < ../3-true/train.true.{language}{param['trained']} " +
               f"> ../3-true_models/bpe.{language}.codes{param['trained']}")

    os.system(command)

def bpe_splitting(name, param, language):
    command = (f"subword-nmt apply-bpe -c ../3-true_models/bpe.{language}.codes{param['trained']}" + 
               f" < ../3-true/{name}.true.{language}{param['trained']} > ../3-true/{name}.bpe.{language}{param['trained']}")

    os.system(command)


def convert_byte(param):
    if param["bpe"] == 0:
        name = "true"
    else:
        name = "bpe"

    command = (f"onmt_preprocess -train_src ../3-true/train.{name}.{param['src_lang']}{param['trained']} " + 
                               f"-train_tgt ../3-true/train.{name}.{param['tgt_lang']}{param['trained']} " + 
                               f"-valid_src ../3-true/val.{name}.{param['src_lang']}{param['trained']} " +  
                               f"-valid_tgt ../3-true/val.{name}.{param['tgt_lang']}{param['trained']} " +  
                               f"-save_data ../4-byte_data/data-{param['src_lang']}-{param['tgt_lang']}{param['trained']}")

    os.system(command)


def train(param):
    command = (f"onmt_train -data ../4-byte_data/data-{param['src_lang']}-{param['tgt_lang']}{param['trained']} " +
                        f"-save_model ../5-trained_models/model-{param['src_lang']}-{param['tgt_lang']}{param['trained']} " +
                        f"-train_steps {param['total_steps']*param['trained']} " +
                        f"-save_checkpoint_steps {param['save_valid']} " +
                        f"-valid_steps {param['save_valid']} " +
                        f"-global_attention {param['attention']} " +
                        f"-input_feed 0 " +
                        f"-dropout 0.1 " +
                        f"-world_size 1 " +
                        f"-gpu_ranks 0 " +
                        f"-layers {param['layers']} " +
                        f"-rnn_size {param['rnn_size']} " +
                        f"-word_vec_size {param['word_vec_size']}")

    if param["transformer"] == "yes":
        command += (" -transformer_ff 2048 -heads 8 " +
                    "-encoder_type transformer -decoder_type transformer -position_encoding "+
                    "-max_generator_batches 2 "+
                    "-batch_size 4096 -batch_type tokens -normalization tokens  -accum_count 2 "+
                    "-optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2 "+
                    "-max_grad_norm 0 -param_init 0  -param_init_glorot " +
                    "-label_smoothing 0.1")
    
    if param["trained"] != 1:
        command += (f" -train_from ../5-trained_models/model-{param['src_lang']}-{param['tgt_lang']}{param['trained'] - 1}_step_{(param['trained']-1)*100000}.pt")

    os.system(command)


def translate(param):
    if param["bpe"] == 0:
        source = (f"../3-true/test.true.{param['src_lang']}1")
    else:
        source = (f"../3-true/test.bpe.{param['src_lang']}1")

    command = (f"onmt_translate -model ../5-trained_models/model-{param['src_lang']}-{param['tgt_lang']}*00000.pt " +  
                                f"-src {source} " +
                                f"-output ../6-raw_pred/pred_{param['tgt_lang']}{param['trained']} " +
                                f"-gpu 0")

    os.system(command)


def detokenize(param):
    if param["bpe"] == 0:
        command = (f"perl ../preprocessing-tools/detokenizer.perl -u -l {param['tgt_lang']} < ../6-raw_pred/pred_{param['tgt_lang']}{param['trained']} "+
                   f"> ../6-detok_pred/pred.detok.{param['tgt_lang']}{param['trained']}")
    else:
        command = (f"sed -r 's/(@@ )|(@@ ?$)//g' < ../6-raw_pred/pred_{param['tgt_lang']}{param['trained']}" + 
        f" | perl ../preprocessing-tools/detokenizer.perl -u -l {param['tgt_lang']} > ../6-detok_pred/pred.detok.{param['tgt_lang']}{param['trained']}")

    os.system(command)

def bleu(param):
    command = (f"perl ../preprocessing-tools/multi-bleu-detok.perl ../0-dataset/test.{param['tgt_lang']}1" + 
               f" < ../6-detok_pred/pred.detok.{param['tgt_lang']}{param['trained']}")

    bleu_score = os.popen(command).read()
    clean_bleu = re.findall("^(.*?),", bleu_score)[0]

    param["last_bleu"] = f'{param["dataset"]["test"]}, {clean_bleu}'
    
    update_json(param, clean_bleu)


def update_json(param, bleu = None):
    # add used dataset
    used = {"Train":  f"{param['dataset']['train']} {param['dataset']['begin_train']}-{param['dataset']['begin_train'] + param['dataset']['train_lines']}",
            "Val":    f'{param["dataset"]["val"]} {param["dataset"]["begin_val"]}-{param["dataset"]["val_lines"] + param["dataset"]["begin_val"]}',
            "Test":   param["dataset"]["test"]
            }

    if param["dataset"]["begin_test"] == param["dataset"]["test_lines"]:
        used["Test"] += " complete"
    else:
        used["Test"] += f' {param["dataset"]["begin_test"]}-{param["dataset"]["test_lines"] + param["dataset"]["begin_test"]}'

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
    
    
    if args.complete == True:

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