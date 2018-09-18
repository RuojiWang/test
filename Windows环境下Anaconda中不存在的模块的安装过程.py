"""
    �����Windows�°�װtorchvision
        һ�������ַ�ʽ���а�װ������ֱ����£�
    Anaconda����İ�װ��ʽ:
        conda install torchvision -c pytorch
    pipָ��İ�װ���̣�
        pip install torchvision
    From source�İ�װ����:
        python setup.py install
        
    �ҿ�������ʹ��Pytorch�ƺ�û�а취����������������⣬��ô��װһ��skorch�����أ�
    skorch��װ����
        pip install -U skorch
        
        ʾ�����룺
        import numpy as np
        from sklearn.datasets import make_classification
        import torch
        from torch import nn
        import torch.nn.functional as F

        from skorch.net import NeuralNetClassifier

        X, y = make_classification(1000, 20, n_informative=10, random_state=0)
        X = X.astype(np.float32)

        class MyModule(nn.Module):
            def __init__(self, num_units=10, nonlin=F.relu):
                super(MyModule, self).__init__()

                self.dense0 = nn.Linear(20, num_units)
                self.nonlin = nonlin
                self.dropout = nn.Dropout(0.5)
                self.dense1 = nn.Linear(num_units, 10)
                self.output = nn.Linear(10, 2)

            def forward(self, X, **kwargs):
                X = self.nonlin(self.dense0(X))
                X = self.dropout(X)
                X = F.relu(self.dense1(X))
                X = F.softmax(self.output(X), dim=-1)
                return X

        net = NeuralNetClassifier(
            MyModule,
            max_epochs=10,
            lr=0.1,
        )

        net.fit(X, y)
        y_proba = net.predict_proba(X)
    
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        pipe = Pipeline([
            ('scale', StandardScaler()),
            ('net', net),
        ])

        pipe.fit(X, y)
        y_proba = pipe.predict_proba(X)
    
        from sklearn.model_selection import GridSearchCV

        params = {
            'lr': [0.01, 0.02],
            'max_epochs': [10, 20],
            'module__num_units': [10, 20],
        }
        gs = GridSearchCV(net, params, refit=False, cv=3, scoring='accuracy')

        gs.fit(X, y)
        print(gs.best_score_, gs.best_params_)
        
        ����skorchʧ�ܽ��������        
            ���г�����ʾModuleNotFoundError: No module named 'skorch.net'; 'skorch' is not a package
            �Ų鵽ԭ������Ϊ�����Ŀ�д���һ������skorch.py�ļ������ļ����޸��Ժ󼴿�   
   
    
    ���ʹ��cuda����ѵ����
        ������ͨ������������ֱ�ӣ�
        x = torch.randn(2, 3)
        x = x.cuda()
    
        ���������磺
        model = MyModel()
        model.cuda()
    
    ����AttributeError: module 'torch' has no attribute 'no_grad'
        ������Ϊtorch.no_gradֻ����Pytorch 0.4���ϰ汾���е�
            ��cmd������ָ�conda install pytorch=0.4.0 -c pytorch����
        
    ʹ��IPDB����Python����
    IPDB��Ipython Debugger������GDB���ƣ���һ�����Ipython��Python���������е��Թ��ߣ����Կ���PDB�������档
        ��װָ�
        IPDB��Python�����������ʽ������ʹ��pip install ipdb�������ɰ�װ��
        ͨ���ڴ��뿪ͷ�����������ֱ���ڴ���ָ��λ�ò���ϵ㡣������ʾ��
        import ipdb
        # some code
        x = 10
        ipdb.set_trace()
        y = 20
        # other code
        ��������ִ����x = 10�������֮��ֹͣ��չ��Ipython�������Ϳ������ɵص����ˡ�
        ��ϸ���Է�������https://xmfbit.github.io/2017/08/21/debugging-with-ipdb/
  
    ���ʹ���Ŵ��㷨���л���ѧϰ��
        ��� ��˼��������������ķ�ʽ��ø��Ž⣬���ⷢ��sklearn-deap���������Ŵ��㷨����ѧϰ��
        һ�������ַ�ʽ���а�װ������ֱ����£�
        pipָ��İ�װ���̣�
            pip install sklearn-deap
        From source�İ�װ���̣���cmd���뵽setup.py����·���£�:
            python setup.py install    
    Example of usage:
        import sklearn.datasets
        import numpy as np
        import random

        data = sklearn.datasets.load_digits()
        X = data["data"]
        y = data["target"]

        from sklearn.svm import SVC
        from sklearn.model_selection import StratifiedKFold

        paramgrid = {"kernel": ["rbf"],
            "C"     : np.logspace(-9, 9, num=25, base=10),
            "gamma" : np.logspace(-9, 9, num=25, base=10)}

        random.seed(1)

        from evolutionary_search import EvolutionaryAlgorithmSearchCV
        cv = EvolutionaryAlgorithmSearchCV(estimator=SVC(),
                                   params=paramgrid,
                                   scoring="accuracy",
                                   cv=StratifiedKFold(n_splits=4),
                                   verbose=1,
                                   population_size=50,
                                   gene_mutation_prob=0.10,
                                   gene_crossover_prob=0.5,
                                   tournament_size=3,
                                   generations_number=5,
                                   n_jobs=4)
        cv.fit(X, y)

    Output:
        Types [1, 2, 2] and maxint [0, 24, 24] detected
        --- Evolve in 625 possible combinations ---
        gen    nevals    avg         min        max
        0      50        0.202404    0.10128    0.962716
        1      26        0.383083    0.10128    0.962716
        2      31        0.575214    0.155259    0.962716
        3      29        0.758308    0.105732    0.976071
        4      22        0.938086    0.158041    0.976071
        5      26        0.934201    0.155259    0.976071
        Best individual is: {'kernel': 'rbf', 'C': 31622.776601683792, 'gamma': 0.001}
        with fitness: 0.976071229827
    
    Example for maximizing just some function:
    
        from evolutionary_search import maximize
    
        def func(x, y, m=1., z=False):
        return m * (np.exp(-(x**2 + y**2)) + float(z))

        param_grid = {'x': [-1., 0., 1.], 'y': [-1., 0., 1.], 'z': [True, False]}
        args = {'m': 1.}
        best_params, best_score, score_results, _, _ = maximize(func, param_grid, args, verbose=False)

    Output:
        best_params = {'x': 0.0, 'y': 0.0, 'z': True}
        best_score  = 2.0
        score_results = (({'x': 1.0, 'y': -1.0, 'z': True}, 1.1353352832366128),
        ({'x': -1.0, 'y': 1.0, 'z': True}, 1.3678794411714423),
        ({'x': 0.0, 'y': 1.0, 'z': True}, 1.3678794411714423),
        ({'x': -1.0, 'y': 0.0, 'z': True}, 1.3678794411714423),
        ({'x': 1.0, 'y': 1.0, 'z': True}, 1.1353352832366128),
        ({'x': 0.0, 'y': 0.0, 'z': False}, 2.0),
        ({'x': -1.0, 'y': -1.0, 'z': False}, 0.36787944117144233),
        ({'x': 1.0, 'y': 0.0, 'z': True}, 1.3678794411714423),
        ({'x': -1.0, 'y': -1.0, 'z': True}, 1.3678794411714423),
        ({'x': 0.0, 'y': -1.0, 'z': False}, 1.3678794411714423),
        ({'x': 1.0, 'y': -1.0, 'z': False}, 1.1353352832366128),
        ({'x': 0.0, 'y': 0.0, 'z': True}, 2.0),
        ({'x': 0.0, 'y': -1.0, 'z': True}, 2.0))
        
    ���е�ʱ��������������쳣��
        RuntimeError: 
        An attempt has been made to start a new process before the
        current process has finished its bootstrapping phase.

        This probably means that you are not using fork to start your
        child processes and you have forgotten to use the proper idiom
        in the main module:

            if __name__ == '__main__':
                freeze_support()
                ...

        The "freeze_support()" line can be omitted if the program
        is not going to be frozen to produce an executable.
        
                ��������Google��������ҳ�����̲������������
        https://github.com/rsteca/sklearn-deap/blob/master/evolutionary_search/cv.py
                ��Ȼ����n_jobs����������������⣬֮ǰPytorch��Tutorial�ͳ���������⡣
        windows����򵥵Ľ���������ǽ�������Ϊ1����������������ڸ��Ӳ����Ƽ���
        
    Eclipse���޷��鿴ĳһ����Դ���룺
                ������������������ԭ���Ǵ����ͻ��������˵�����ͬʱ��װ�������汾��A�����ת��ĳһ�����Ķ���֮����
    
    �����Windows�°�װadvisor
    Run the advisor server.
        pip install -r ./requirements.txt
        ./manage.py runserver 0.0.0.0:8000
    pipָ��İ�װ���̣�
        pip install advisor_clients
        
        client = AdvisorClient()
    Example of usage:
        # Create the study
        study_configuration = {
            "goal": "MAXIMIZE",
            "maxTrials": 5,
            "maxParallelTrials": 1,
            "params": [
                    {
                        "parameterName": "hidden1",
                        "type": "INTEGER",
                        "minValue": 40,
                        "maxValue": 400,
                        "scallingType": "LINEAR"
                    }
            ]
        }
        study = client.create_study("Study", study_configuration)

        # Get suggested trials
        trials = client.get_suggestions(study, 3)

        # Complete the trial
        client.complete_trial(trial, trial_metrics)

    ʹ�ð�װ��Ҷ˹�Ż���
        ��Ҷ˹�Ż��Ŀ�������֮�࣬��Ϊ��û���ҵ���ؿ�ıȽϱ��棬�ҿ��ܻᰤ����һ�µİɣ�
        �����֮���ٿ���ʱ�������������ѡ�񣬺���������ѡ��hyperopt�ǶԵģ���Ϊ���ֽ׶β�����ʹ��tensorflow
        ���ԾͲ�̫����ʹ��ZhuSuan�ˣ�����hyperoptʵ����sklearn���������пⶼ��sklearnΪ����,��ǳ�����BO�����Ͳ�����        
            ����ǰ�װhyperopt��
        pip install hyperopt
            ���������ֵı�����ô����cmd��ִ�����е����
        pip3 install networkx==1.11
        
            �����ǰ�װBayesian Optimization��������˾���û����ĺ��þͲ��ÿ�����
        pip install bayesian-optimization
        
    ����ĳ���Ѿ���װ�Ŀ⣺
        �ٴ������䰲װ����ɣ�������Ҫ��skorch���µ����°汾������������ָ��ɣ�
        pip install -U skorch
        
    ��װhyperopt-sklearn:
        ��Ϊhyperopt�����кܶ��ȱ�ݣ�����˵��Ҫʵ�ֽ�ѵ���õ�ģ������Ԥ���Ȼ�����Լ�д����
        ��Ȼ�Ҹ��˾��ÿ϶������ܹ�����Щ���ܵĴ���д�ıȽ�native������ֱ�Ӱ�װhysklearn�ܲ���ǿѽ
            ������windows�°�װGit����Ϊ�ҵ��İ�װ��ʽ����Ҫ��Git���أ�����֮��һ·next���в�����װ·�����뵽������������
            Ȼ����cmd������뵽Git���ص�����Ŀ¼������python setup.py install����
    
    ������ת�����⣺
    eclipse�о�������Ҫ��ת�鿴������������ඨ���ʱ�򣬴󲿷�ʱ������ƶ��������ϵ�������ķ��򼴿ɣ�
        ������������������ÿ�ζ��ܹ���Ч�������ʱ��ֻ��Ҫ�������õ����鿴�����ϣ�Ȼ�󰴼��̵�ctrl�����ɣ�
        ������װ֮�����еĺ������඼�����ڱ��أ���֧����������ת�����°�װ�ļ��Ѿ���ɾ����
        ��֮ǰ����Ϊhyperopt�еĴ����޷���ת�����Ծ��أ��Ҿ�˵���˵������Խ��뵽��ת��������Ӧ�û��б�İ취����
  
      �������pip:
    python -m pip install --upgrade pip
    
    ��ΰ�װTPOT�Ŀ⣺
        ��������ָ�
    conda install numpy scipy scikit-learn pandas
    pip install deap update_checker tqdm stopit
    pip install xgboost
    pip install scikit-mdr skrebate
    pip install tpot
    
    ��ΰ�װfeaturetools:
        ��������ָ�
    conda install -c featuretools featuretools
    
    ��ΰ�װauto keras:
    pip install autokeras
    ��װautokerasʧ�ܣ� 
    Could not find a version that satisfies the requirement torch==0.4.0 (from autokeras)
        �����ַ�����˾���Ľ������https://qiita.com/ruteshi_SI_shiteru/items/93ffd1161ff219d286fc
        ��׵ķ����ǽ�https://github.com/hiteshn97/autokeras/ ��������Ȼ��ʹ��python setup.py install��װ
        ��Ȼ��ʾ��ִ�����������ʱ����ʾ��һЩ��ֵĶ�������auto keras�ƺ�ȷʵ�ǰ�װ��������

  Keras��ܴ:
    GPU�汾
    pip install --upgrade tensorflow-gpu
    CPU�汾
    pip install --upgrade tensorflow
    Keras ��װ
    pip install keras -U --pre
  
  pydot��graphviz�İ�װ
        ��pip install graphviz��װgraphviz
        ��������Ҫ��װgraphViz��windows��������Ҫ���������վ����
        https://graphviz.gitlab.io/_pages/Download/Download_windows.html
        Ȼ����������վ��for windows�Ľ���������������¶���(����msi��װ��Ȼ��ͬʱ���û�������ϵͳ����������Ӷ���)
        https://stackoverflow.com/questions/35064304/runtimeerror-make-sure-the-graphviz-executables-are-on-your-systems-path-aft
            �����Ҫͬʱ���û����������D:\Graphviz2.38\bin;����ϵͳ����������ӣ�D:\Graphviz2.38\bin\dot.exe;�������������޷�����ʹ��
            ���Ǻ���ֱ��conda install graphviz�Ϳ����ˣ�ֱ��condaӦ�ò��аɣ�conda��ô�������ϵͳ�����ͻ��������������ͬʱ����������������޷����С�
            ͨ��conda��װ�Ľ����ǵ���������Ľӿڶ��ѣ����������������Ǳ�����windows���氲װ���������޷����С�
        ������������Eclipse�����ò���
        Ȼ��pip install pydot==1.1.0
        ����˳��һ��Ҫ��ȷ����pydot������1.1.0������ָ�����������⡣����������Ҳ���а���
        ���������AttributeError: 'ImageClassifier' object has no attribute 'layers'�Ĵ���
        �����Ϊ����һ�� tensorflow�汾��keras�汾��ƥ������⣬����tensorflow����keras�İ汾����Ѫ��û���á�
    cmd������conda install tensorflow==1.10.0�����ǰ�װ�̶��汾���﷨����������һ���汾�������Կ�����Щ�ǺϷ��汾���޷�������������ˡ�
        ʾ�������и���û��plot_model֮��Ķ�������ű��������Ͳ�֧��plot_model��
        
    auto_ml�İ�װ
            ����pip install auto_ml�����¶���
            
"""