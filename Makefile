#!/usr/bin/env sh
ai_model=vit
chip_architect=M3D

N_tier=4
N_stack=1
N_tile=256
N_pe=64
N_crossbar=8
xbar_size=128

quant_weight=8
quant_act=8

freq_computing=1
fclk_noc=1
tsvPitch=10
W2d=32
voltage=0.5

placement_method=2
routing_method=2
percent_router=0.5
router_times_scale=1

sim:
	python3 main.py --ai_model $(ai_model) --chip_architect $(chip_architect) --xbar_size $(xbar_size) --N_tile $(N_tile) --N_crossbar $(N_crossbar) --N_pe $(N_pe) --N_stack $(N_stack) --quant_weight $(quant_weight) --quant_act $(quant_act) --freq_computing $(freq_computing) --fclk_noc $(fclk_noc) --tsvPitch $(tsvPitch) --N_tier $(N_tier) --voltage $(voltage) --placement_method $(placement_method) --routing_method $(routing_method) --percent_router $(percent_router) --W2d $(W2d) --router_times_scale $(router_times_scale) --thermal
	@echo "Simulation Done!"