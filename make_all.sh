#! /bin/bash

function check_answer(){
	flag=0
	while [ $flag -eq 0 ]
	do
		read answer
		if [ x$answer = x"n" ]
		then
			flag=1
			return 0
		elif [ x$answer = x"y" ]
		then
			flag=1
			return 1
		else
			echo "??? Please give 'y' or 'n'. Use 'Ctrl-C' to exit."
		fi
	done
}

function get_j(){
	idx=0; for op in $@; do ((idx=$idx+1)); if [ x$op = x"-j" ]; then break; fi; done; ((idx=$idx+1)); j=`eval echo '$'$idx`
}

base_folder=`pwd`
root_folder=${base_folder}/spipy
env_name="spipy-v3"

# get opts
SKIP_COMPILE=0
SKIP_CONDAENV=0
while getopts ":xeh" opt; do
	case $opt in
		x)
			SKIP_COMPILE=1
			;;
		e)
			SKIP_CONDAENV=1
			;;
		h)
			echo "Use : ./make_all.sh [-x] (Do not compile MPI C codes)"
			echo "                    [-e] (Do not check conda environment)"
			echo "                    [-h] (help info)"
			exit 0
			;;
		\?)
			echo "[Error] Invalid option : $OPTARG . Exit."
			exit 1
			;;
	esac
done

# check whether there is anaconda installed
curr_env_path=`which python`
curr_env_path=${curr_env_path%/*bin/python*}
a='anaconda'
b='miniconda'
if [[ $curr_env_path =~ $a ]] || [[ $curr_env_path =~ $b ]]
then
	echo "[Info] Root folder is $root_folder"
else
	echo "[Error] Use anaconda/miniconda and switch to base environment, please. Exit."
	exit 1
fi

# check whether anaconda is switched to base env
a='envs'
if [[ $curr_env_path =~ $a ]]
then
	echo "[Error] Please switch conda to base environment !"
	exit 0
fi

# check python version
py_version=`conda list | grep "^python "`
a='3.*.*'
if [[ $py_version =~ $a ]]
then
	echo "==> Anaconda version authorized"
else
	echo "[Error] Your python version is $py_version. Exit."
	exit 1
fi


# decide your system
sys=`uname`

if [ $sys != "Linux" ] && [ $sys != "Darwin" ]
then
	echo "[Error] I can't recognize your system. Exit."
	exit 1
fi

if [ $sys = "Darwin" ] && [ $SKIP_COMPILE -ne 1 ]
then
	echo "[Warning] Since now I didn't add any support on compiling EMC module on MacOS, '-x' option will be added automatically. Continue ? (y/n)"
	check_answer
	if [ $? -eq 0 ]; then
		exit 1
	fi
	SKIP_COMPILE=1
fi


# decide mpicc
echo "==> Operating system authorized"
if [ $SKIP_COMPILE -eq 1 ]; then
	echo "[Info] Skip compiling merge.emc module."
fi

if [ $SKIP_COMPILE -eq 0 ]; then
	echo "==> Authorizing MPI"
	# decide your gcc
	if [ $sys = "Darwin" ]
	then
		nowgcc=`which gcc`
		echo "[Warning] I need openmp/MPI/GSL support. Do you want to use current gcc? : $nowgcc [y/n]"
		check_answer
		if [ $? -eq 1 ]
		then
			mygcc=gcc
		else
			echo "Give me your specific gcc path : "
			read mygcc
		fi
	fi
	# reject conda mpi
	if [ $sys = "Linux" ]
	then
		nowmpicc=`which mpicc`
		nowmpirun=`which mpirun`
		if [ -z "$nowmpicc" ]; then echo "[Error] There is no mpi detected, exit";exit 0;fi
		if [ $nowmpicc = "${curr_env_path}/bin/mpicc" ] || [ $nowmpirun = "${curr_env_path}/bin/mpirun" ]
		then
			echo "[Warning] The current mpicc is $nowmpicc"
			echo "   Make sure whether it contains GSL and openMP libraries."
			echo "   Or give me another mpicc absolute path (type 'n' to to use current one) : "
			read mympicc
			if [ $mympicc = "n" ]
			then
				mympicc=$nowmpicc
			fi
		else
			mympicc=$nowmpicc
		fi
		# record mpirun
		mympirun=`dirname $mympicc`/mpirun
	fi
fi


# start compiling ...
echo "==> Compile image/bhtsne_source"
cd $root_folder/image/bhtsne_source
g++ sptree.cpp tsne.cpp tsne_main.cpp -o bh_tsne -O2
if [ $? -ne 0 ];then echo $?; exit 1;fi
chmod u+x bh_tsne

echo "==> Compile image/qlist_dir"
cd $root_folder/image/qlist_dir
g++ gen_quat.cpp -o gen_quat -O1
if [ $? -ne 0 ];then echo $?; exit 1;fi
chmod u+x gen_quat

cd $root_folder/phase
chmod u+x ./template_2d/new_project ./template_3d/new_project

if [ $SKIP_COMPILE -eq 0 ]; then

	echo "==> Compile merge/template_emc/src"
	cd $root_folder/merge/template_emc/src
	chmod u+x compile.sh ../new_project
	if [ $sys = "Linux" ]
	then
		$mympicc -fopenmp recon.c setup.c max.c quat.c interp.c -o emc_LINUX -I ./ -lgsl -lgslcblas -lm -O3
		if [ $? -ne 0 ];then echo $?; exit 1;fi
		chmod u+x emc_LINUX
	elif [ $sys = "Darwin" ]
	then
		$mygcc -fopenmp recon.c setup.c max.c quat.c interp.c -o emc_MAC -I ./ -lgsl -lgslcblas -lm -O3 -lmpi
		if [ $? -ne 0 ];then echo $?; exit 1;fi
		chmod u+x emc_MAC
	fi

fi

if [ $? -ne 0 ];then echo "Command failed. Exit"; exit 1;fi


# extract test data
echo "==> Extracting test data"
cd $root_folder/../demo
tar -xzf test.tgz
tar -xzf analyse/analyse.tgz -C ./analyse/
tar -xzf image/image.tgz -C ./image/
tar -xzf merge/merge.tgz -C ./merge/
tar -xzf phase/phase.tgz -C ./phase/
tar -xzf simulate/simulate.tgz -C ./simulate/

if [ $? -ne 0 ];then echo "Command failed. Exit"; exit 1;fi


# check conda env
if [ $SKIP_CONDAENV -eq 0 ]; then
	echo "==> Checking conda environment"

	all_envs=`conda env list | grep ${env_name}`
	if [ -z "$all_envs" ]
	then
		echo "[Warning] The coming procedure will create a new conda environment named '${env_name}'. Continue ? (y/n)"
		check_answer
		if [ $? -eq 0 ]
		then
			exit 1
		fi
		conda env create -f ${root_folder}/../environment.yaml -n ${env_name}
	else
		echo "[Info] The target conda environment '${env_name}' already exists, we will use it."
		conda env update -f ${root_folder}/../environment.yaml -n ${env_name}
	fi
else
	echo "[INFO] skip checking conda environment"
fi

if [ $? -ne 0 ];then echo "Command failed. Exit"; exit 1;fi

# install to conda env, make soft link
echo "==> Install spipy to conda env"
python_path=`find ${curr_env_path}/envs/${env_name}/lib -maxdepth 1 -name "python3.*"`
if [ ! -d ${python_path} ]
then
	echo "I can't find an unique python in your anaconda environment."
	echo "ERR_PATH = ${python_path}"
	exit 1
fi

if [ ! -d "${python_path}/site-packages/spipy" ]
then
	ln -fs $root_folder ${python_path}/site-packages/spipy
else
	echo "[Warning] spipy is already in python3.*/site-packages. Over-write it? [y/n]"
	check_answer
	if [ $? -eq 1 ]
	then
		rm ${python_path}/site-packages/spipy
		ln -fs $root_folder ${python_path}/site-packages/spipy
	else
		echo "Skip."
	fi
fi

if [ $? -ne 0 ];then echo "Command failed. Exit"; exit 1;fi

# write info.py :
INFO=$root_folder/info.py
if [ -f "$INFO" ]; then
	rm -rf $INFO
fi
touch $INFO
# version
echo "VERSION = 3.2" >> $INFO
# mympirun
if [ $SKIP_COMPILE -eq 0 ]; then
	echo "EMC_MPI = '$mympirun'" >> $INFO
fi


# generate shell files for running scripts
echo "==> Generate execuated files"
cd $base_folder/scripts
if [ -d $base_folder/bin ]
then
	rm -rf $base_folder/bin
fi
mkdir $base_folder/bin
# generate files
env_bin_path=${curr_env_path}/envs/${env_name}/bin
for scripts_f in `ls *.py`;
do
	bin_f=${base_folder}/bin/${scripts_f%.py*}
	scripts_fn=${base_folder}/scripts/${scripts_f}
	echo "# !/bin/bash" >> $bin_f
	echo "" >> $bin_f
	echo "idx=0; for op in \$@; do ((idx=\$idx+1)); if [ x\$op = x\"-j\" ]; then break; fi; done; ((idx=\$idx+1))" >> $bin_f
	echo "" >> $bin_f
	echo "if (( \$idx <= \$# )); then j=\`eval echo '\$'{\$idx}\`; else j=0; fi" >> $bin_f
	echo "" >> $bin_f
	echo "if (( \$j >= 1 )); then ${env_bin_path}/mpirun -np \$j ${env_bin_path}/python ${scripts_fn} \$@; else ${env_bin_path}/python ${scripts_fn} \$@; fi" >> $bin_f
	chmod u+x $bin_f
done

if [ $? -ne 0 ];then echo "Command failed. Exit"; exit 1;fi


# complete
cd $root_folder
echo "==> Complete!"
