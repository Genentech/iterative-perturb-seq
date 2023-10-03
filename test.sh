cd projects/iterpert_repo

for i in TypiClust 
#BatchBALD Random TypiClust
do
python run.py --strategy $i
done

#bsub -J run_replogle -q long -n 8 -R "rusage[mem=4GB]" -gpu num=1 -o ./log/run_replogle.o%J -e ./log/run_replogle.e%J /home/huangk28/projects/iterpert_repo/test.sh 