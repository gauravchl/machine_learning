#!/usr/bin/env bash

# Use exported model using 

path=./models/1.0.2
tag=serve
signature=predict_output

saved_model_cli run --dir $path --tag_set $tag --signature_def $signature --input_exprs 'x=4'
