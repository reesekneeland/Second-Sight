# Mental Imagery, Imagery Trials

# python src/second_sight.py --idx "0,1,2,3,4,5,6,7,8,9,10,11" --subjects 2 --device cuda:3 --output output/mi_imagery/ --noae --log --mi
# python src/second_sight.py --idx "0,1,2,3,4,5,6,7,8,9,10,11" --subjects 5 --device cuda:2 --output output/mi_imagery/ --noae --log --mi
# python src/second_sight.py --idx "0,1,2,3,4,5,6,7,8,9,10,11" --subjects 7 --device cuda:2 --output output/mi_imagery/ --noae --log --mi

# Mental Imagery Vision Trials

# python src/second_sight.py --idx "0,1,2,3,4,5,6,7,8,9,10,11" --subjects 1 --device cuda:2 --output output/mi_vision/ --noae --log --mivis
# python src/second_sight.py --idx "10,11" --subjects 2 --device cuda:3 --output output/mi_vision/ --noae --log --mivis
# python src/second_sight.py --idx "4,5,6,7,8,9,10,11" --subjects 5 --device cuda:1 --output output/mi_vision/ --noae --log --mivis
python src/second_sight.py --idx "4,5,6,7,8,9,10,11" --subjects 7 --device cuda:1 --output output/mi_vision/ --noae --log --mivis
cd ../SS_release_test/Second-Sight/
python src/stochastic_search_statistics.py