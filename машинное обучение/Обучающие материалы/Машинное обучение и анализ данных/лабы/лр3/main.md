{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Деревья решений и метод ближайших соседей в задачах классификации. Метрики оценки качества обучения."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "При подготовке \"тетради\" использовались материалы:\n",
    "\n",
    "Открытый курс машинного обучения. Тема 3. Классификация, деревья решений и метод ближайших соседей:\n",
    "- https://habr.com/ru/company/ods/blog/322534/\n",
    "    \n",
    "Метрики в задачах машинного обучения\n",
    "\n",
    "- https://habr.com/ru/company/ods/blog/328372/\n",
    "\n",
    "Отличия LabelEncoder и OneHotEncoder в SciKit Learn\n",
    "\n",
    "- https://habr.com/ru/post/456294/\n",
    "\n",
    "Крайне рекомендуется ознакомиться с представленными материалами до или после ознакомления с \"тетрадью\"."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "На данном занятии будет рассмотрена задача классификации. Классификация – это отнесение объекта к одной из категорий на основании его признаков. Классификация является одной из двух задач обчения с учителем (класс задач, в которых у нас есть обучающая выборка данных). Вторая задача обучения с учителем - задача регрессии. Регрессия – прогнозирование количественного признака объекта на основании прочих его признаков.\n",
    "В рамках занятия будет рассмотрена задча бинарной классификации, т.е. будет всего 2 класса: 0 и 1. Задачей будет выступать вопрос: получит ли конкретный спортсмен медаль (любой ценности) на олимпиаде? Попытаемся ответить на этот вопрос с помощью таких алгоритмов машинного обучения как дерево решений и метод ближайших соседей.\n",
    "\n",
    "Импорт необходимых библиотек:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Путь к данным:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "path = '../../Dataset/athlete_events.csv'"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Считывание данных:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = pd.read_csv(path)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Заполнение пустых (nan) значений:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "data.fillna(0, inplace = True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Посмотрим на \"голову\" набора данных:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ID</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>Height</th>\n",
       "      <th>Weight</th>\n",
       "      <th>Team</th>\n",
       "      <th>NOC</th>\n",
       "      <th>Games</th>\n",
       "      <th>Year</th>\n",
       "      <th>Season</th>\n",
       "      <th>City</th>\n",
       "      <th>Sport</th>\n",
       "      <th>Event</th>\n",
       "      <th>Medal</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>A Dijiang</td>\n",
       "      <td>M</td>\n",
       "      <td>24.0</td>\n",
       "      <td>180.0</td>\n",
       "      <td>80.0</td>\n",
       "      <td>China</td>\n",
       "      <td>CHN</td>\n",
       "      <td>1992 Summer</td>\n",
       "      <td>1992</td>\n",
       "      <td>Summer</td>\n",
       "      <td>Barcelona</td>\n",
       "      <td>Basketball</td>\n",
       "      <td>Basketball Men's Basketball</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>A Lamusi</td>\n",
       "      <td>M</td>\n",
       "      <td>23.0</td>\n",
       "      <td>170.0</td>\n",
       "      <td>60.0</td>\n",
       "      <td>China</td>\n",
       "      <td>CHN</td>\n",
       "      <td>2012 Summer</td>\n",
       "      <td>2012</td>\n",
       "      <td>Summer</td>\n",
       "      <td>London</td>\n",
       "      <td>Judo</td>\n",
       "      <td>Judo Men's Extra-Lightweight</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>Gunnar Nielsen Aaby</td>\n",
       "      <td>M</td>\n",
       "      <td>24.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Denmark</td>\n",
       "      <td>DEN</td>\n",
       "      <td>1920 Summer</td>\n",
       "      <td>1920</td>\n",
       "      <td>Summer</td>\n",
       "      <td>Antwerpen</td>\n",
       "      <td>Football</td>\n",
       "      <td>Football Men's Football</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>Edgar Lindenau Aabye</td>\n",
       "      <td>M</td>\n",
       "      <td>34.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Denmark/Sweden</td>\n",
       "      <td>DEN</td>\n",
       "      <td>1900 Summer</td>\n",
       "      <td>1900</td>\n",
       "      <td>Summer</td>\n",
       "      <td>Paris</td>\n",
       "      <td>Tug-Of-War</td>\n",
       "      <td>Tug-Of-War Men's Tug-Of-War</td>\n",
       "      <td>Gold</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>Christine Jacoba Aaftink</td>\n",
       "      <td>F</td>\n",
       "      <td>21.0</td>\n",
       "      <td>185.0</td>\n",
       "      <td>82.0</td>\n",
       "      <td>Netherlands</td>\n",
       "      <td>NED</td>\n",
       "      <td>1988 Winter</td>\n",
       "      <td>1988</td>\n",
       "      <td>Winter</td>\n",
       "      <td>Calgary</td>\n",
       "      <td>Speed Skating</td>\n",
       "      <td>Speed Skating Women's 500 metres</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   ID                      Name Sex   Age  Height  Weight            Team  \\\n",
       "0   1                 A Dijiang   M  24.0   180.0    80.0           China   \n",
       "1   2                  A Lamusi   M  23.0   170.0    60.0           China   \n",
       "2   3       Gunnar Nielsen Aaby   M  24.0     0.0     0.0         Denmark   \n",
       "3   4      Edgar Lindenau Aabye   M  34.0     0.0     0.0  Denmark/Sweden   \n",
       "4   5  Christine Jacoba Aaftink   F  21.0   185.0    82.0     Netherlands   \n",
       "\n",
       "   NOC        Games  Year  Season       City          Sport  \\\n",
       "0  CHN  1992 Summer  1992  Summer  Barcelona     Basketball   \n",
       "1  CHN  2012 Summer  2012  Summer     London           Judo   \n",
       "2  DEN  1920 Summer  1920  Summer  Antwerpen       Football   \n",
       "3  DEN  1900 Summer  1900  Summer      Paris     Tug-Of-War   \n",
       "4  NED  1988 Winter  1988  Winter    Calgary  Speed Skating   \n",
       "\n",
       "                              Event Medal  \n",
       "0       Basketball Men's Basketball     0  \n",
       "1      Judo Men's Extra-Lightweight     0  \n",
       "2           Football Men's Football     0  \n",
       "3       Tug-Of-War Men's Tug-Of-War  Gold  \n",
       "4  Speed Skating Women's 500 metres     0  "
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Для решения задачи классификации необходимо определить целевой признак - факт получения медали. Для этого текущий признак \"Medal\" следует преобразовать: значения \"Gold\", \"Silver\", \"Bronze\" заменить на 1, а значения 0 оставить без изменения. Это преобразование осуществляет следующий код:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "data['Medal'] = data['Medal'].apply(lambda x: 1 if x != 0 else 0)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Еще раз посмотрев на \"голову\" набора данных можем убедиться, что значения заменены в соответствии с указанным требованием."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>ID</th>\n",
       "      <th>Name</th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>Height</th>\n",
       "      <th>Weight</th>\n",
       "      <th>Team</th>\n",
       "      <th>NOC</th>\n",
       "      <th>Games</th>\n",
       "      <th>Year</th>\n",
       "      <th>Season</th>\n",
       "      <th>City</th>\n",
       "      <th>Sport</th>\n",
       "      <th>Event</th>\n",
       "      <th>Medal</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>A Dijiang</td>\n",
       "      <td>M</td>\n",
       "      <td>24.0</td>\n",
       "      <td>180.0</td>\n",
       "      <td>80.0</td>\n",
       "      <td>China</td>\n",
       "      <td>CHN</td>\n",
       "      <td>1992 Summer</td>\n",
       "      <td>1992</td>\n",
       "      <td>Summer</td>\n",
       "      <td>Barcelona</td>\n",
       "      <td>Basketball</td>\n",
       "      <td>Basketball Men's Basketball</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>A Lamusi</td>\n",
       "      <td>M</td>\n",
       "      <td>23.0</td>\n",
       "      <td>170.0</td>\n",
       "      <td>60.0</td>\n",
       "      <td>China</td>\n",
       "      <td>CHN</td>\n",
       "      <td>2012 Summer</td>\n",
       "      <td>2012</td>\n",
       "      <td>Summer</td>\n",
       "      <td>London</td>\n",
       "      <td>Judo</td>\n",
       "      <td>Judo Men's Extra-Lightweight</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>Gunnar Nielsen Aaby</td>\n",
       "      <td>M</td>\n",
       "      <td>24.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Denmark</td>\n",
       "      <td>DEN</td>\n",
       "      <td>1920 Summer</td>\n",
       "      <td>1920</td>\n",
       "      <td>Summer</td>\n",
       "      <td>Antwerpen</td>\n",
       "      <td>Football</td>\n",
       "      <td>Football Men's Football</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>Edgar Lindenau Aabye</td>\n",
       "      <td>M</td>\n",
       "      <td>34.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Denmark/Sweden</td>\n",
       "      <td>DEN</td>\n",
       "      <td>1900 Summer</td>\n",
       "      <td>1900</td>\n",
       "      <td>Summer</td>\n",
       "      <td>Paris</td>\n",
       "      <td>Tug-Of-War</td>\n",
       "      <td>Tug-Of-War Men's Tug-Of-War</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>Christine Jacoba Aaftink</td>\n",
       "      <td>F</td>\n",
       "      <td>21.0</td>\n",
       "      <td>185.0</td>\n",
       "      <td>82.0</td>\n",
       "      <td>Netherlands</td>\n",
       "      <td>NED</td>\n",
       "      <td>1988 Winter</td>\n",
       "      <td>1988</td>\n",
       "      <td>Winter</td>\n",
       "      <td>Calgary</td>\n",
       "      <td>Speed Skating</td>\n",
       "      <td>Speed Skating Women's 500 metres</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   ID                      Name Sex   Age  Height  Weight            Team  \\\n",
       "0   1                 A Dijiang   M  24.0   180.0    80.0           China   \n",
       "1   2                  A Lamusi   M  23.0   170.0    60.0           China   \n",
       "2   3       Gunnar Nielsen Aaby   M  24.0     0.0     0.0         Denmark   \n",
       "3   4      Edgar Lindenau Aabye   M  34.0     0.0     0.0  Denmark/Sweden   \n",
       "4   5  Christine Jacoba Aaftink   F  21.0   185.0    82.0     Netherlands   \n",
       "\n",
       "   NOC        Games  Year  Season       City          Sport  \\\n",
       "0  CHN  1992 Summer  1992  Summer  Barcelona     Basketball   \n",
       "1  CHN  2012 Summer  2012  Summer     London           Judo   \n",
       "2  DEN  1920 Summer  1920  Summer  Antwerpen       Football   \n",
       "3  DEN  1900 Summer  1900  Summer      Paris     Tug-Of-War   \n",
       "4  NED  1988 Winter  1988  Winter    Calgary  Speed Skating   \n",
       "\n",
       "                              Event  Medal  \n",
       "0       Basketball Men's Basketball      0  \n",
       "1      Judo Men's Extra-Lightweight      0  \n",
       "2           Football Men's Football      0  \n",
       "3       Tug-Of-War Men's Tug-Of-War      1  \n",
       "4  Speed Skating Women's 500 metres      0  "
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Для того, чтобы воспользоваться рассматриваемыми алгоритмами, также следует преобразовать строковые признаки в числовые. Это необходимо:\n",
    "- деревьям решений для возможности сравнения значений между собой;\n",
    "- методу ближайших соседей для возможности подсчета расстояний между анализируемыми данными;\n",
    "\n",
    "Посмотрим какие признаки на данный момент имеют тип \"object\" или \"string\" и требуеют преобразования:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 271116 entries, 0 to 271115\n",
      "Data columns (total 15 columns):\n",
      " #   Column  Non-Null Count   Dtype  \n",
      "---  ------  --------------   -----  \n",
      " 0   ID      271116 non-null  int64  \n",
      " 1   Name    271116 non-null  object \n",
      " 2   Sex     271116 non-null  object \n",
      " 3   Age     271116 non-null  float64\n",
      " 4   Height  271116 non-null  float64\n",
      " 5   Weight  271116 non-null  float64\n",
      " 6   Team    271116 non-null  object \n",
      " 7   NOC     271116 non-null  object \n",
      " 8   Games   271116 non-null  object \n",
      " 9   Year    271116 non-null  int64  \n",
      " 10  Season  271116 non-null  object \n",
      " 11  City    271116 non-null  object \n",
      " 12  Sport   271116 non-null  object \n",
      " 13  Event   271116 non-null  object \n",
      " 14  Medal   271116 non-null  int64  \n",
      "dtypes: float64(3), int64(3), object(9)\n",
      "memory usage: 31.0+ MB\n"
     ]
    }
   ],
   "source": [
    "data.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Признак ID использоваться не будет - это уникальный идентификатор спортсмена, который в значительной степени соотносится с получением медали. В реальной задаче этот признак мог бы стать одним из ключевых. Но, т.к. это учебная задача - пока следует обойтись без него.\n",
    "\n",
    "Признак имени также не будет использоваться в рамках решения задачи классификации (можете потом сами его добавить и посмотреть как он повлияет на точность).\n",
    "\n",
    "Для решения задачи классификации будут использованы следующие признаки: 'Sex', 'Age', 'Height', 'Weight', 'NOC', 'Season', 'Sport', 'Event'. Признаки 'Age', 'Height', 'Weight' уже имеют числовые значения и их преобразование не требуется.\n",
    "Т.к. признаки 'Sex' и 'Season' содержать всего по 2 уникальных значения - их можно преобразовать с использованием lambda:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "data['Sex'] = data['Sex'].apply(lambda x: 1 if x == 'M' else 0)\n",
    "data['Season'] = data['Season'].apply(lambda x: 1 if x == 'Summer' else 0)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Признаки 'NOC', 'Sport', 'Event' содержат гораздо большее количество уникальных значений, поэтому их преобразование вручную может потребовать много времени. Для автоматического преобразования строкового признака в числовой можно возпользоваться кодировщиком LabelEncoder из библиотеки sklearn.preprocessing. LabelEncoder закодирует каждое уникальное значение столбца целым числом и заменит все значения столбца числами.\n",
    "\n",
    "<b>Следует сделать очень важное заменчание.</b> LabelEncoder переведет признаки 'NOC', 'Sport', 'Event' в набор чисел. Но это всего лишь категориальные данные, и между числами на самом деле нет никакой связи. Т.е. например спорт > 50 не будет нести какой-либо смысловой нагрузки. Для того, чтобы правильно закодировать категориальные признаки, следует воспользоваться OneHotEncoder или pandas.get_dummies При помощи этих инструментов для каждого категориального признака будет создан свой столбец. Но, чтобы упростить данное занятие и сосредоточиться непосредственно на алгоритмах, описанные преобразования будут рассмотрены в следующем занятии.\n",
    "\n",
    "Преобразование с исользованием LabelEncoder:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import LabelEncoder\n",
    "\n",
    "le = LabelEncoder()\n",
    "\n",
    "data['NOC'] = le.fit_transform(data['NOC'])\n",
    "data['Sport'] = le.fit_transform(data['Sport'])\n",
    "data['Event'] = le.fit_transform(data['Event'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Выделим из набора данных целевой признак:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [],
   "source": [
    "target = data['Medal']"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Выделим только те колонки, которые будут признаками для классификации:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "cols = ['Sex', 'Age', 'Height', 'Weight', 'NOC', 'Season', 'Sport', 'Event']\n",
    "data = data[cols]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "И посмотрим на получившийся кадр данных:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Sex</th>\n",
       "      <th>Age</th>\n",
       "      <th>Height</th>\n",
       "      <th>Weight</th>\n",
       "      <th>NOC</th>\n",
       "      <th>Season</th>\n",
       "      <th>Sport</th>\n",
       "      <th>Event</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>24.0</td>\n",
       "      <td>180.0</td>\n",
       "      <td>80.0</td>\n",
       "      <td>41</td>\n",
       "      <td>1</td>\n",
       "      <td>8</td>\n",
       "      <td>159</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>23.0</td>\n",
       "      <td>170.0</td>\n",
       "      <td>60.0</td>\n",
       "      <td>41</td>\n",
       "      <td>1</td>\n",
       "      <td>32</td>\n",
       "      <td>397</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>1</td>\n",
       "      <td>24.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>55</td>\n",
       "      <td>1</td>\n",
       "      <td>24</td>\n",
       "      <td>348</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>1</td>\n",
       "      <td>34.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>55</td>\n",
       "      <td>1</td>\n",
       "      <td>61</td>\n",
       "      <td>709</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>21.0</td>\n",
       "      <td>185.0</td>\n",
       "      <td>82.0</td>\n",
       "      <td>145</td>\n",
       "      <td>0</td>\n",
       "      <td>53</td>\n",
       "      <td>622</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Sex   Age  Height  Weight  NOC  Season  Sport  Event\n",
       "0    1  24.0   180.0    80.0   41       1      8    159\n",
       "1    1  23.0   170.0    60.0   41       1     32    397\n",
       "2    1  24.0     0.0     0.0   55       1     24    348\n",
       "3    1  34.0     0.0     0.0   55       1     61    709\n",
       "4    0  21.0   185.0    82.0  145       0     53    622"
      ]
     },
     "execution_count": 25,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Учитывая, что все признаки теперь числовые - можно перейти к созданию моделей. В первую очередь требуется импортировать необходимые библиотеки:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Деревья решений\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "#Разбивка набора данных на обучающую и тестовую выборки\n",
    "from sklearn.model_selection import train_test_split"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Весь набор данных будет разделен на обучающую и тестовую выборки в соотношении 70:30 с использованием train_test_split:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_holdout, y_train, y_holdout = train_test_split(data, target, test_size=0.2, random_state=42)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Посмотрим на размер получившихся выборок:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "((216892, 8), (54224, 8), (216892,), (54224,))"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train.shape, X_holdout.shape, y_train.shape, y_holdout.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "97952     0\n",
       "151066    0\n",
       "13248     0\n",
       "5803      0\n",
       "249601    0\n",
       "         ..\n",
       "119879    0\n",
       "259178    0\n",
       "131932    0\n",
       "146867    0\n",
       "121958    0\n",
       "Name: Medal, Length: 216892, dtype: int64"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "y_train"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Создание дерева классификаии с использованием критерия качества разбиения - энтропия и максимальной глубиной дерева = 5."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "clf_tree = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=17)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Обучение дерева:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=17)"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clf_tree.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Проведем процедуру \"предсказания\" полчения медали:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 0, ..., 0, 0, 0], dtype=int64)"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "pred = clf_tree.predict(X_holdout)\n",
    "pred"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Теперь требуется оценить качество полученных предсказаний. В первую очередь следует попробовать accuracy_score из библиотеки sklearn.metrics."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import accuracy_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8548059899675421"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "accuracy_score(y_holdout, pred)"
   ]
  },
  {
   "attachments": {
    "image.png": {
     "image/png": "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"
    }
   },
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Полученный результат ~ 85% точных предсказаний. Однако, учитывая что одно дерево решений - это очень слабый алгоритм, данный показатель выглядит как минимум странно. Для понимания почему получено такое высокое значение следует обратиться к формуле расчета показателя accuracy:\n",
    "\n",
    "![image.png](attachment:image.png)\n",
    "\n",
    "где:\n",
    "- TP - верные предсказания по классу 1;\n",
    "- FP - неверные предсказания по классу 1;\n",
    "- TN - верные предсказания по классу 0;\n",
    "- FN - неверные предсказания по классу 0;\n",
    "\n",
    "Общее количество значений класса 1 в наборе данных:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "39783"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "target.sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Общее количество строк в наборе данных:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "271116"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.shape[0]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Соотношение количества записей с классом 1 к общему количеству записей:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 105,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.14673792767671404"
      ]
     },
     "execution_count": 105,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "target.sum() / data.shape[0]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Т.е. этот показатель говорит о том, что даже если отнести все предстказания к классу 0, то accuracy будет равен:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 106,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.853262072323286"
      ]
     },
     "execution_count": 106,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "(0 + target.shape[0] - target.sum()) / (0 + target.shape[0] - target.sum() + 0 + target.sum())"
   ]
  },
  {
   "attachments": {
    "image.png": {
     "image/png": "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"
    }
   },
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Значение accuracy, полученное с помощью дерева решений, всего на 0.003657942678200188 больше, чем если бы все предсказания были отнесены к классу 0. Такая ситуация возникает из-за значительного дисбаланса классов (0 намного больше, чем 1). И в подобных ситуациях метрика accuracy не показательна.\n",
    "\n",
    "Более показательными в данном случае будут метрики precision, recall и агрегированные критерии на их основе.\n",
    "![image.png](attachment:image.png)"
   ]
  },
  {
   "attachments": {
    "image.png": {
     "image/png": "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"
    }
   },
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![image.png](attachment:image.png)\n",
    "\n",
    "Как видно из формул и изображения:\n",
    "- precision - это соотношение правильных предсказаний класса 1 по отношению к сумме правильных и не правльных предсказаний класса 1;\n",
    "- recall - это соотношение правильных предсказаний класса 1 по отношению к сумме правильных предсказаний класса 1 и неправильных предсказаний класса 0;\n",
    "\n",
    "Инструмент classification_report позволяет сразу оценить precision, recall и f1-score сделанных предсказаний. (Подробно про f1-score можно прочитать в материале - https://habr.com/ru/company/ods/blog/328372/)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import classification_report"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.86      1.00      0.92     46290\n",
      "           1       0.58      0.03      0.06      7934\n",
      "\n",
      "    accuracy                           0.85     54224\n",
      "   macro avg       0.72      0.51      0.49     54224\n",
      "weighted avg       0.82      0.85      0.79     54224\n",
      "\n"
     ]
    }
   ],
   "source": [
    "report = classification_report(y_holdout, clf_tree.predict(X_holdout))\n",
    "print (report)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Как видно из полученной таблицы - доля верных предсказаний класса 1 по метрике recall - вего 0.02%. Попытаемся улучшить точность путем увеличения глубины дерева."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [],
   "source": [
    "clf_tree = DecisionTreeClassifier(criterion='entropy', max_depth=20, random_state=17)\n",
    "clf_tree.fit(X_train, y_train)\n",
    "pred = clf_tree.predict(X_holdout)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.90      0.94      0.92     46290\n",
      "           1       0.56      0.42      0.48      7934\n",
      "\n",
      "    accuracy                           0.87     54224\n",
      "   macro avg       0.73      0.68      0.70     54224\n",
      "weighted avg       0.85      0.87      0.86     54224\n",
      "\n"
     ]
    }
   ],
   "source": [
    "report = classification_report(y_holdout, clf_tree.predict(X_holdout))\n",
    "print (report)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Как видно, при глубине дерева = 20, recall = 0.4, что уже значительно лучше. А f1-score, который объединяет precission и recall, возрос до 0.47, что является одним из лучших значений, которое можно достичь с применением 1 дерева решений.\n",
    "Можно также попробовать использовать другой критерий качества разбиения в узлах - Неопределенность Джини."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 179,
   "metadata": {},
   "outputs": [],
   "source": [
    "clf_tree = DecisionTreeClassifier(criterion='gini', max_depth=20, random_state=17)\n",
    "clf_tree.fit(X_train, y_train)\n",
    "pred = clf_tree.predict(X_holdout)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 180,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.90      0.94      0.92     69460\n",
      "           1       0.53      0.42      0.47     11875\n",
      "\n",
      "    accuracy                           0.86     81335\n",
      "   macro avg       0.72      0.68      0.69     81335\n",
      "weighted avg       0.85      0.86      0.85     81335\n",
      "\n"
     ]
    }
   ],
   "source": [
    "report = classification_report(y_holdout, clf_tree.predict(X_holdout))\n",
    "print (report)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "При этом f1-score остался неизменным, но изменились значения precision и recall.\n",
    "\n",
    "Для автоматического подбора параметров модели можно воспользоваться инструментом GridSearchCV. Суть того, как работает GridSearchCV: для каждой уникальной пары значений параметров max_depth и max_features будет проведена 5-кратная кросс-валидация и выберется лучшее сочетание параметров.\n",
    "\n",
    "Параметр max_depth будет изменяться в диапазоне от 1 до 31 с шагом 5. Параметр max_features (количество используемых признаков) будет изменяться в диапазоне от 1 до 7 с шагом 1."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import GridSearchCV, cross_val_score\n",
    "from sklearn.metrics import recall_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [],
   "source": [
    "tree_params = {'max_depth': range(1,32,5),\n",
    "'max_features': range(4,7)}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [],
   "source": [
    "tree_grid = GridSearchCV(clf_tree, tree_params, scoring = 'recall',\n",
    "cv=5, n_jobs=-1,\n",
    "verbose=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitting 5 folds for each of 21 candidates, totalling 105 fits\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "GridSearchCV(cv=5,\n",
       "             estimator=DecisionTreeClassifier(criterion='entropy', max_depth=20,\n",
       "                                              random_state=17),\n",
       "             n_jobs=-1,\n",
       "             param_grid={'max_depth': range(1, 32, 5),\n",
       "                         'max_features': range(4, 7)},\n",
       "             scoring='recall', verbose=True)"
      ]
     },
     "execution_count": 40,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tree_grid.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'max_depth': 31, 'max_features': 6}"
      ]
     },
     "execution_count": 41,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tree_grid.best_params_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.47084651839648145"
      ]
     },
     "execution_count": 42,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "tree_grid.best_score_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.4844971010839425"
      ]
     },
     "execution_count": 43,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "recall_score(y_holdout, tree_grid.predict(X_holdout))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "В результате найдены лучшие параметры модели: глубина дерева - 31, количество признаков - 6. Оценка на кросс-валидации ~ 0.47, оценка recall на отложенной выборке ~ 0.48.\n",
    "\n",
    "Полученно дерево решений можно визуализировать с применением export_graphviz. Предварительно необходимо переобучить дерево с малой глубиной, чтобы можно было рассмотреть все дерево."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=17)"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "clf_tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=17)\n",
    "clf_tree.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "\"dot\" ­Ґ пў«пҐвбп ў­гваҐ­­Ґ© Ё«Ё ў­Ґи­Ґ©\n",
      "Є®¬ ­¤®©, ЁбЇ®«­пҐ¬®© Їа®Ја ¬¬®© Ё«Ё Ї ЄҐв­л¬ д ©«®¬.\n"
     ]
    }
   ],
   "source": [
    "# используем .dot формат для визуализации дерева\n",
    "from sklearn.tree import export_graphviz\n",
    "export_graphviz(clf_tree, feature_names=['Sex', 'Age', 'Height', 'Weight', 'NOC', 'Season', 'Sport', 'Event'], \n",
    "out_file='small_tree.dot', filled=True)\n",
    "# для этого понадобится библиотека pydot (pip install pydot)\n",
    "!dot -Tpng 'small_tree.dot' -o '../../img/small_tree.png'"
   ]
  },
  {
   "attachments": {
    "image.png": {
     "image/png": "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"
    }
   },
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![image.png](attachment:image.png)\n",
    "Если у Вас не отображается дерево. Оно автоматически сохранено в формате .dot в папке с данным notebook. Данные из файла (который можно открыть текстовым редактором), следует вставить в форму на сайте https://dreampuf.github.io/GraphvizOnline/ . В правой части окна дерево автоматически построится."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Попробуем рассмотренную задачу классификации решить с применением метода ближайших соседей.\n",
    "\n",
    "Основные параметры класса sklearn.neighbors.KNeighborsClassifier:\n",
    "- weights: \"uniform\" (все веса равны), \"distance\" (вес обратно пропорционален расстоянию до тестового примера) или другая определенная пользователем функция\n",
    "- algorithm (опционально): \"brute\", \"ball_tree\", \"KD_tree\", или \"auto\". В первом случае ближайшие соседи для каждого тестового примера считаются перебором обучающей выборки. Во втором и третьем — расстояние между примерами хранятся в дереве, что ускоряет нахождение ближайших соседей. В случае указания параметра \"auto\" подходящий способ нахождения соседей будет выбран автоматически на основе обучающей выборки.\n",
    "- leaf_size (опционально): порог переключения на полный перебор в случае выбора BallTree или KDTree для нахождения соседей\n",
    "- metric: \"minkowski\", \"manhattan\", \"euclidean\", \"chebyshev\" и другие"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "\n",
    "knn = KNeighborsClassifier(n_neighbors=10)\n",
    "\n",
    "knn.fit(X_train, y_train)\n",
    "\n",
    "pred_knn = knn.predict(X_holdout)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.88      0.97      0.93     46290\n",
      "           1       0.63      0.25      0.36      7934\n",
      "\n",
      "    accuracy                           0.87     54224\n",
      "   macro avg       0.76      0.61      0.64     54224\n",
      "weighted avg       0.85      0.87      0.84     54224\n",
      "\n"
     ]
    }
   ],
   "source": [
    "report = classification_report(y_holdout, pred_knn)\n",
    "print (report)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Как видно из результата - при значении параметра n_neighbors=10 (количество объектов по которым ищется ближайший), значение recall = 0.24, а f-1-score = 0.34."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYAAAAEICAYAAABWJCMKAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMSwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/YYfK9AAAACXBIWXMAAAsTAAALEwEAmpwYAAA9QUlEQVR4nO3deXhV1bn48e+biZAQkpCEAEmAEBJmREBQAVFRC+h1qr2CU22dp4q1t8VeW9HeOrS3ansd0ToUB+rPkVYcqkUBGWRQ5gQSxjCeBMIUQobz/v7YO3AIgZyEk5wk5/08T56cs/bea68dZb17r7X2WqKqGGOMCT1hwS6AMcaY4LAAYIwxIcoCgDHGhCgLAMYYE6IsABhjTIiyAGCMMSHKAoAxfhKRjSJyQQOOO1dEChujTCc5Z1sR+YeI7BWR/+fH/qtE5Fz38xQReaOxy2iCzwKAaVRupXlIRA74/DzThOdv1pWZGxxURJ6rkT5XRG50P9/o7vPLGvsUVlfatbgKSAWSVPVHdZVDVfup6lcNuATTglkAME3hP1S1nc/P3cEuUDNzELheRLqfZJ/dwC9FJM7PPLsBa1W18lQLZ1ovCwAmaEQkS0T+LSLFIlIkIm+KSILP9l+JyFYR2S8ieSIyRkQ6iUipiCT57DdYRDwiElkj/7HAr4Gr3SePZW76T0RkjZvvehG5zeeYZBH5p4iUiMhuEZkjIsf9OxGRPiKyQUQmNuC6fyYiq0Uk3U0qAV4DHjrJYWuA+cDP/cj/YeC3HL3um/z4W9favCUi0SLyhntciYgsEpFUvy/WNGsWAEwwCfAY0AXoA2QAUwBEpBdwN3CGqsYBPwA2quoO4CvgP33yuR6YrqoVvpmr6qfAo8Df3SeP09xNu4BLgPbAT4CnRGSwu+1+oBBIwWlC+TVwzHwp7r6fAfeo6tv1umCR3wI3AqNV1bdf4PfAD93rPpHfAJNEpMPJzqGqD3Hsdf+Vk/yt6/BjIN7dPwm4HTjkx3GmBbAAYJrCh+7dY/XPLQCqmq+q/1LVw6rqAZ4ERrvHVAFtgL4iEqmqG1W1wN32OnAdgIiEAxOBaf4WRlU/VtUCdXwNfA6McjdXAJ2Bbqpaoapz9NgJs0YBM4AbVPWf9fgbiIg8CVwEnOder2+ZdgAvAI+cpNzfA/8CflWP81Yfe7K/9clU4FT8PVW1SlWXqOq++p7fNE8WAExTuFxVE3x+XgIQkVQRme428+wD3gCSwamwgEk4d6m73P26uPl9hBMYMoELgb2q+q2/hRGRcSKywG3iKQHGV58X+COQD3zuNg9NrnH47cC8BnSYJgC3Ao+p6t4T7PME8AMROe0E28Fp2rmjvs0wJ/tb12EaztPOdBHZJiJ/qNnUZlouCwAmmB7FaV4ZoKrtce7qpXqjqr6lqiNxOjQVp4JEVcuAd9z9r+fkd/81m2/aAO8B/wukqmoCMLP6vKq6X1XvV9UewKXAz0VkjE8WtwNdReSpel7rHpxmp1dFZEStBVUtBp4GfnfCi1HNBd4H/rue5z/p3/ok56tQ1YdVtS9wtnsNN9Tz3KaZsgBggikOOADsFZE04L+qN4hILxE5362wy3Danb0+x/4Npy39Uk4eAHYC3X06cqNwmpY8QKWIjMNplqk+7yUi0lNEBNiL0xTle979wFjgHBF5vD4X6z41XAu8LyLDTrDbkzgVbZ+TZPUwTt9FQj1Of8K/9cmIyHkiMsBtatuH0yTkreMw00JYADBN4R9y7HsAH7jpDwODcSraj3HubKu1AR4HioAdQEfggeqNqvoNTkW0VFU3neTc1S9BFYvIUlXdD/wM5wliD3ANTpt+tWzgC5zKcj7wnKrO8s1QVUtwmp7GicjvAETkExH5dV1/CFX9F/BT928yuJbt+4A/ACfs6FXVDThBL7au8/k42d/6ZDoB7+JU/muAr6lHf4tp3sQWhDEtlYj8G3hLVV8OdlmMaYksAJgWSUTOwBkRk+He1Rtj6smagEyLIyKv4zTTTLLK35iGsycAY4wJUfYEYIwxISoi2AWoj+TkZO3evXuwi2GMMS3KkiVLilQ1pWZ6iwoA3bt3Z/HixcEuhjHGtCgiUutQaWsCMsaYEGUBwBhjQpQFAGOMCVEWAIwxJkRZADDGmBBlAcAYY0KUBQBjjAlRLeo9AGNMiPJ6Yc8G2PYdFOc7aWHhEBYB4v4OC68lLQLCwo5+PpJeW1rN433Tq/f1Oe7IvmEgda6t0yxZADDGNC++lf3272Hb97B9ORw+0UqazYDUDBbhtQehkwamGnnUTLvsWYhoE9BiWwAwxgRPXZV9eBSk9oMBP4TOg6DLIEjp41SK3krnR6vcz1XuzwnSjqT7k+Y9+vlInrWkeatqnKtmWqVzjceUySet5vkrD9dI88m37hU8680CgDGmaTS0so+Iqj2/sCicFT5NQ1kAMMYEXqAre9MoLAAYY06NVfYtlgUAY4z/rLJvVSwANKbiAlj4Ilz0u4D33hvT6FRh93qr7FsxvwKAiIwF/gyEAy+r6uM1tncFXgcS3H0mq+pMd9tA4EWgPeAFzlDVMhH5CugMHHKzuUhVd53qBTUrq96Hb1+E1L4w5MZgl8aYE7PKPiTVGQBEJBx4FrgQKAQWicgMVV3ts9uDwDuq+ryI9AVmAt1FJAJ4A7heVZeJSBJQ4XPctaraeld48eQ5v7/5M5x+vTOW15hg87ey738ldDndKvtWzJ8ngGFAvqquBxCR6cBlgG8AUJw7fIB4YJv7+SJguaouA1DV4kAUusXw5EKb9s4/tjUzoN8VwS6RCTVW2ZuT8CcApAFbfL4XAsNr7DMF+FxE7gFigQvc9BxAReQzIAWYrqp/8DnuVRGpAt4D/kdVtebJReRW4FaArl27+lHcZsJbBUXrYMhPYN1nMPdp6Ht5i31l3LQAVtmbegpUJ/BE4DVV/ZOInAVME5H+bv4jgTOAUuBLEVmiql/iNP9sFZE4nABwPfC3mhmr6lRgKsDQoUOPCxDNVslmqCxz2v879oZ/3Avrv4Ks84JdMtNaHNoDBbNg21Kr7E2D+BMAtgIZPt/T3TRfNwFjAVR1vohEA8k4TwuzVbUIQERmAoOBL1V1q7v/fhF5C6ep6bgA0GJVt/+n9IbOp8Gsx2DuUxYAzKnZswnyZkLux7BpnjNlgFX2poH8CQCLgGwRycSp+CcA19TYZzMwBnhNRPoA0YAH+Az4pYjEAOXAaOApt3M4QVWLRCQSuAT4IhAX1Gx4cp3fyTnOENCz7oR//Ra2LoG0IcEtm2k5VJ3mnNyZTsW/c6WTntIbRtwLvcY5o3KssjcNUGcAUNVKEbkbpzIPB15R1VUi8giwWFVnAPcDL4nIfTgdwje67fl7RORJnCCiwExV/VhEYoHP3Mo/HKfyf6kxLjBoPHnQrhO0TXC+D/kJzP6T0xdw9bRglsw0d5XlsHGOU+HnfQL7tjpTDmcMh4v+B3qNh6SsYJfStAJ+9QG4Y/pn1kj7rc/n1cCIExz7Bs5QUN+0g0Drvg325EJKr6Pfo9vDsJthzpNO53BydvDKZpqfsr2w7l9Opb/uX3B4H0S0hZ5j4LxfQ85YiE0OdilNgHi9yv6ySnaXlrP7YDkl7u89peXsKa1gT/XngxXsLi2npLSCBQ+cT0R4YNfwsjeBG4MqFK2FQdcemz78dpj/rPNewGXPBKdspvnYW+jc4ed+DBvngrcCYpKh76XQ+xLocS5Etg12KU0dvF5lX1mFW4E7lbdTaZez+2DFCSt37wmGtESGCwkxUXSIiSIxNpLsju1IjI2iokqJCPCrRBYAGsO+rVB+4NgnAIB2HeH062DJ685dXfsuwSmfCQ5Vpw0/dybkfQzblznpST3hzDug98WQfoa9MBhEXq+y91CFW1kfvQPfU2vl7qSV1FGZJ8ZE0SE2ioSYSHJS2/lU7lEkxkSSGOt+dyv8dm0ikCYaLm4BoDFUdwDXDAAAZ90Ni1+BBc857bmmdauqcEbr5M10Kv69mwFxKvoLpkCviyElJ9ilbJWqK3Pfu/Hqu+/aK/WTV+ZR4WEkxkY6FXVMFL07tSchJtKt3KPo4LOtQ6xTwcdGhTdZZd4QFgAagzsEdHN4V16ZsYoHxvemTfWzW4dM6HclLH4VRt0PbRODWFDTKA7vh/wvnAp/3WdO+35EtNOkc84vnJE77ToGu5QtTpVX2bmvjG0lh9x2c9+7c7eCP3LnXs7eQxV+VeYdYqPo06n9MZV79R17h9jqO/PmX5k3hAWAxuDJg5gkPlp7mNfmbaRXpzgmDvN5i3nkJFj5Lix6Gc75r6AV0wTQvu3uqJ2ZsGE2VJVD2w7OHX7v8ZB1PkTFBruUzZqq4jlwmC27D1G4p5TCPc7vLbsPsWVPKdtKDlFRdXyNHhURdkyTSp/O7UmMifRJizrSzFJdqce0wsq8ISwANAZPHqT0Jm/nfgBe+LqAHw1JP9qD32kA9LwQFrwAZ94FUTFBLKxpEFXYtcZpy8+d6byNC5CYCcNudYZqZgyHcPsnVk1V2VNacaRSL9xTyhafz4V7DnG40nvMMcntokhPjGFAWjzjB3QmPbEtaQltSYptQ2KsU5m3jbTKvKHs/85AU3X6APpfybp1B0iIiWRTcSkfr9jOZYPSju438j54bTx8/yYMuyV45TX+q6qELQuOduLu2eikpw2B83/jdOKm9A7p+Z72lVWwZbdTmVf/9q3wD5ZXHbN/fNtIMjq0JSc1jvN7dySjQwzpiW3JSIwhLbEtMVFWRTUm++sG2oFdUFZCZYccCjwHuHlUD75cs5PnZhXwHwO7EBbmVg7dzob0YTDvL85LYnan2DwdPgAF/3aadtZ+Bod2O1MvZI523sTNGQftOwe7lE2mtLzymMr9yO89pWzZXcq+sspj9m/XJsKp0DvEcHbPJNITY8hIbEt6YgzpHdrSPjoySFdiwAJA4BU5HcA72nSj0qv06RxHr07tuO/vy/h37i4u6Jvq7CfiPAVMnwirPoCBPwpioc0x9u+EtZ84d/rrv4KqwxCdADk/cJp2eo6BNnHBLmWjKKuoYmvJoeMq98I9hyjcXUrxwfJj9o+ODDtSqQ/umkhGh7bud+dOPiEm0ppnmjELAIHmjgBaXdEF2Ep2xzhyUtvxp8/X8sysfMb06Xj0H0TOWKfJYO5TMOCqkG46CKrqF/dyP3bu9AsXAwoJXWHoT51O3K5nQXjLv1utqPKyvaTMrdR92+KdCn/X/sPH7B8ZLqQlOHfwF/XrdORuvrqZJrldlFXwLZgFgEDz5EKbeFbsjSY8TOiREktEeBi3j87iwQ9XMn99MWdnua/0h4XBiEnw4e3O6/85FwW16CHFWwVbvj3aibu7wEnvPMh5Sa/XeGeGzRZWuVV5lR37yijc7VTqhTU6WbfvPXTM0MjwMKFzfDTpiW0ZnZNytHJ3f6fGRR9ttjStjgWAQPPkQUoOa3cdoHtSDNGRzvj/q4ak8+cv1/HcrIKjAQCcO/9//4/zFGABoHGVl8L6WU6Fv/ZTKC2CsEjIHOW8idtrPMSn1Z1PM1BZ5WVZ4V4Wbihmc3HpkWaamkMlRSA1LpqMDm0ZltnBaX/3uYPvHB8d8PllTMthASDQPHmQcxFr1x2gT+ej7cTRkeHcPDKTxz7JZdmWEk7LSHA2hEfC2XfDp5Nh8wLoemZwyt1aHfA4lX3eTGfxlMpD0CYesi90mnZ6XgDR8cEupV82FR9k9roi5q7zMK+gmP1uh2tyuzakJ7ZlYHoCFw/o7LTBu23xXRKij76EaEwNFgACqXQ3HNxFRYccNhYf5LJBx871c+2Z3Xh2Vj7PfZXPi9cPPbph8A3w9RPOVNHXTG/aMrdGRflHm3a2LAQU2qfD4Oudu/xuI1rE/Pl7SyuYV1DEnPwi5qzzsGX3IQDSEtpyycDOjOyZwtlZSSTGNv9rMc2TBYBAcjuAt0Z2QxVyUo8dKdKuTQQ3jsjkL1+uY93O/WRXb4+KdWYK/eox2LnaWUbS1M/hA84sq6s/dDp0wXnhbvSvnDv9TgObfXt+RZWX77eUMGethzn5RSzbUoJXnf9vzuyRxC2jejAqO4XuSTHW8WoCwgJAILmTwK2p6AwUHRcAAH5ydndenrOe578q4MmrBx3dMOxWpwL75s9w5YtNU97WYsNs+OguKNkCmefAGTc78+0kdK372CBSVTYUHWTOuiLmrCtiwfpiDhyuJEzgtIwE7j4/m1HZyQzKSCDS2ulNI/ArAIjIWODPOKt3vayqj9fY3hV4HUhw95nsLiKDiAwEXgTaA17gDFUtE5EhwGtAW5zFZu51VxFruYrWQmQM3+9rR1T4bronHT/FQ2JsFNcM68qr8zZy34U5ZHRw94npAENuhG+nwvn/3ewrr2ah/CB8McX5m3XoAT/9tNn3oew5WM43BUXMdSv9rSVOs07XDjFcNqgLo7KTOSsrmfi2LX/IqWn+6gwAIhIOPAtciLPI+yIRmeGuAlbtQeAdVX1eRPriVOjd3bV/3wCuV9VlIpIEVLjHPA/cAix09x8LfBKg6woOTy4k57B210GyOrY74eiKm0f14G/zN/Hi7AL+5/IBRzecdZdTmc1/FsY90USFbqE2fgMf3ekskj78Dhjz22Y5p1J5pZelm/cwZ52HueuKWL51L6oQFx3B2VlJ3HFuFqOyk+mWZBPFmabnzxPAMCBfVdcDiMh04DLANwAozh0+QDywzf18EbBcVZcBqGqxm0dnoL2qLnC//w24nBYfAPKg+yjWrj3AGd1PPM1zp/hofjgknXcWF/Kz87Pp2D7a2RCfDgOvdhaMOeeXEJvURAVvQcpL4ctHYOELkNgNbvwYute6GmlQqCoFngPMXlvE3HynWae0vIrwMOH0jAQmjclhZHYyp6XH2/BLE3T+BIA0YIvP90JgeI19pgCfi8g9QCxwgZueA6iIfAakANNV9Q9unoU18mwZA7BPpGwf7NvK4cSebC05xDWpJ2/CuX10D/6+aDN/nbuBB8b3ObphxL3OBHHfvui8kGSO2rwAPrwDdq93+kwumNIsplguPnCYbwqKmbPWw9z8IrbvLQMgMzmWq4akM7JnMmdmJdm8N6bZCVQn8ETgNVX9k4icBUwTkf5u/iOBM4BS4EsRWQLs9TdjEbkVuBWga9dm3C7ujjzZEu6UsVctHcC+uiXFcsnALryxYBN3nJtFQow7lC+llzOH/MIX4eyfQZt2jVrsFqHikPOy3PxnISEDfvwPp7M3SA5XVrFk4x5nTH6+h5Vb9wHOzJYjeibxs+wURvZMPtq/Y0wz5U8A2Apk+HxPd9N83YTTho+qzheRaCAZ585+tqoWAYjITGAwTr9Aeh154uY3FZgKMHTo0ObbSewOAV1T1QXYR69OdU8Wdse5WcxYto3X523i3guyj24YeZ8zjn3p606/QCjb8q1z11+cD0NvggsfafKgqKqs3XmAOes8zFlXxMINxZRVeIkIEwZ3S+QXF+UwMjuFAWnxhNu0CaYF8ScALAKyRSQTp5KeAFxTY5/NwBjgNRHpA0QDHuAz4JciEgOUA6OBp1R1u4jsE5EzcTqBbwD+LxAXFDSeXAhvw3f72xMTdZC0hLZ1HtKnc3su6NORV+dt4OZRmcS2cf9zZJwB3UbCvGfgjFtaxEtLAVdRBrN+D/OfgfZpcP2HkHVek53es/8w3+QXMdvtvK2eJC0rJZYJZ3RlVHYyw3sk0a6NjaQ2LVed//eqaqWI3I1TmYcDr6jqKhF5BFisqjOA+4GXROQ+nA7hG90hnXtE5EmcIKLATFX92M36To4OA/2E1tABnNSTvF2HyE6N83sCrTvP68mVz83j7W83c/OoHkc3jLwP3vwhrPh/cPq1jVToZqpwiXPXX5TnDI298HcQ3b7Ow05FWUUVizbuZu66ImavK2LNdqdZJzEmkhE9kzknO4WR2cl08SOwG9NSSEsaej906FBdvHhxsItRu6cHQtoQhuZdy3m9Uvjjj07z+9CJUxewvugAs3953tF5W1ThhVHOXPR3LnRmDm3tKg/DV4/DN09DXGe49C/OXD2NQFVZs30/c/OdZp1vN+zmcKWXqPAwhnRLZFROMqN6ptCvS3ubDdO0eCKyRFWH1ky359dAKC+Fks2U9r2aogOH/Wr/93XXeT257q8LeX/p1qOLx4s4i8e/d5MzkVmfSwJf7uZk61L48E7wrIHTr4MfPBrwSdp27StjzjpneOacdUUUHXCadXJS23Hdmd0YmZ3M8MwOtgyhCRn2f3ogFK8DlC0RTuVd2xQQJzOiZxKnpccfv3h838vh37+DuU866822xvlfKsth9h9gzpPQriNc+64zU2cAHCqvYuGG4iNv3ebt3A84C42P6JnMKHe0Tqf46ICcz5iWxgJAILgjgHKrugBl9X4CEBHuPK8nt01bcuzi8eERzlDQj38OG+c689a3JtuXwQd3wK5VcNo1MPZRaHviF+jq4vUqq7fvc+fW8bB44x7Kq7xERYQxrHsHrhycxsjsZPp0smYdY8ACQGB4ckHCWbIvkfi2RXSMa1PvLC7sk0p2x3bHLx4/6FqnXXzuU60nAFSWw5w/wZz/hZgkmPh36DW2wdmpKh+v2M7//HMNO/Y5L2H17hTHjSO6M7JnMsMyOxxZmMcYc5QFgEDw5EFSFms8ZeSktmvQVL1hYcId52bx83dqLB4fGQ1n3u5Mf7B9GXT2v3O5Wdqxwhnhs2OFM+3F2MedifAaaPveQ/zmw5V8sWYXA9Li+dW4XozomUzHOGvWMaYuITC0pAl48tCUXuTt2F/v9n9f/3FaF9IT2/LMrHyOGZ019CaIinOmim6pqirg6z/A1PNg/06Y8BZcObXBlb/Xq7yxYBMXPjmbuflF/Pf4Pnxw59lccXq6Vf7G+MkCwKmqPAy713OwfU/2lVXWu/3fV2R4GLeNzuL7LSXMX198dEPbBDjjp7DqA2cenJZm52p4eYzzYlffy+CuhU6ndgMVeA4wYeoCHvxwJadlxPPZpHO45ZweNrmaMfVk/2JOVXEBaBVbwp3ZMk7lCQDgR0PSSW7Xhue/Kjh2w5l3QlgEzGtBL0xXVTpt/S+eA3u3wn/+Da76a4Pv+iuqvDw7K59xf55D7o59/OGqgbxx03CbStmYBrI+gFNV5M4BVNkF8J5yAIiODOeWUbUsHh/XCQZdA9+9CaMnQ1zqqZW7se3Kddr6ty11hrNe/CeITW5wdssLS/jlu8vJ3bGfiwd05qFL+1pTjzGnyJ4ATpUnDxAW7U8iJa4NHQKwQPe1Z3ajfXQEz32Vf+yGs38G3gpY+Pwpn6PReKucxe1fHAV7NsJVr8J/vt7gyv9QeRW//3g1lz/7DXtKy5l6/RCevXawVf7GBIA9AZwqTy4kdme1p5yc1MDMUnnCxeOTspw29EV/deYKCvCbsqfMs9ZZpatwEfS+BC55ynm5q4G+yS/igfdXsHl3KROHdWXyuN62VKIxAWRPAKfKk4cm57B254FTbv7x9ZOzuxMTFX58X8CISXB4Hyx+JWDnOmXeKqdv4oWRzrTNP/wrXP1Ggyv/vaUV/Nf/W8a1Ly8kPEyYfuuZPHblAKv8jQkwCwCnoqoSivPZH9eTQxVVdS4CUx/Vi8d/tGwbW3aXHt3QZRBknQ8LnnemTA62onx4dRx8/qAzcdudC2HAVQ2atkJVmbliO2Oe/Jr3v9vKHedm8cm9ozizhy2NaUxjsABwKvZshKpytkS4I4BOYQhobW4e1YNwEV6cXctTwIGdsOztgJ6vXrxemP8cvDDCaQa7YipMeLPBndM79pZx67Ql3PnmUjrFt+Gju0bwq7G97Q1eYxqR9QGcCk8uAKsrugCQ3TGwK1U5i8enHb94fOY50GWw82LY4BsgrIkryeIC+Ohu2DwPcsbCJU9D+84NysrrVaYv2sJjM9dQXuXlgXG9uWlkpo3pN6YJ2L+yU+EGgEUHkklLaEtcIyz6fds5WVRWefnr3A1HE0WcTuA9G2D1RwE/5wl5vc5axS+MhJ2r4PLnYeL0Blf+6z0HmPjSAn79wQr6pzkvdN02Ossqf2OaiD0BnApPHrRPZ7mnKmAjgGrqnnx08fg7z+1JfIwbZHpfAknZziRx/a5o/Kmid29w7vo3zXXa+v/jLxCf1qCsKqq8vDRnPU9/sY42EWE88cMB/OfQjAbNoWSMaTi/brVEZKyI5IlIvohMrmV7VxGZJSLfichyERnvpncXkUMi8r3784LPMV+5eVZva/h4wWApysOb0ov1noMBb//3dce5WRwsr+L1+RuPJoaFwYh7YcdyKPh3o50brxcWvQzPj3DOdekzzpz9Daz8VxTu5bJnvuEPn+YxpndHvvz5aK4+o6tV/sYEQZ1PACISDjwLXAgUAotEZIaqrvbZ7UHgHVV9XkT6AjOB7u62AlUddILsr1XVZrrGYx28XvCsZV+/YZRXeQM6Aqim6sXjX/lmAzeN9Fk8fuB/wqxHnaeAnmMCf+I9m2DG3bBhNvQ4Dy79P0jIaFBWh8qrePqLtbw0Zz3J7drwwnVDGNu/U4ALbIypD3+eAIYB+aq6XlXLgenAZTX2UaB61e54YFvgithM7d0MlYcCNgdQXe48ryclpRW8/e3mo4kRbeCsu2DjHGch9UBRhcWvwvNnO0s1XvI0XP9Bgyv/eflFjP3zbF6cvZ6rz8jgXz8fbZW/Mc2APwEgDdji873QTfM1BbhORApx7v7v8dmW6TYNfS0iNVc0edVt/vmNtLQ2AHcVsNUVXQgT6BngEUA1De6ayFk9knhpznoOV1Yd3TDkxxCdAN88FZgTlWyBaVfAPydB2mC4cz4M/UmD+hj2llbwq3eXc83LCxHgrVuG89iVA+2FLmOaiUANt5gIvKaq6cB4YJqIhAHbga6qejrwc+AtEal+UrhWVQcAo9yf62vLWERuFZHFIrLY4/EEqLgB4AaAbw8k0y0ptknGq991Xk927jvM+0u3Hk1sEwfDboE1/3SmYmgoVVj6N3juLNjyrTN52/UfQULXBmX3yYrtXPDU17y7tJDbRvfg00nncHZWwyeDM8YEnj8BYCvg++yf7qb5ugl4B0BV5wPRQLKqHlbVYjd9CVAA5Ljft7q/9wNv4TQ1HUdVp6rqUFUdmpKS4u91NT5PHrRL5bsiabQRQDX5Lh5fWeU9umH47RARDfMauGDM3q3w5lUw4x7nTeM758EZNzsdzfW0c18Zt01bzB1vLiWlnfNC1wPj+tgLXcY0Q/78C18EZItIpohEAROAGTX22QyMARCRPjgBwCMiKW4nMiLSA8gG1otIhIgku+mRwCXAykBcUJPx5FKV3IuNRQcbtQPYl4hwx7k92VRcyscrth/dEJsMg6+HZX93KnN/qTrTSz93FmyaB+P+CDfMgMTu9S6bqvL2t5u54Mmv+SrPw6/G9uaju0fQP62ZTVhnjDmizgCgqpXA3cBnwBqc0T6rROQREbnU3e1+4BYRWQa8DdyozpqG5wDLReR74F3gdlXdDbQBPhOR5cD3OE8ULwX0yhqTKnjy2BuTiVcDPwXEyVzUN5We7uLxXq/PspFn3Q3qhQXP+ZfRvu3w1tXO7J2p/eCOb2D4rQ26699YdJCJLy3ggfdX0K9Lez6ddA53nJtFpL3QZUyz5teLYKo6E6dz1zfttz6fVwMjajnuPeC9WtIPAkPqW9hmY/92KN/PlginfbyxRwD5CgsT7qxt8fjEbs4kbItfhVH3n3jVLVVY/nf45JdQWe4syj7stgZV/JVVXl6as4Gnv1hLVEQYj105gKuHZhAW1rL6840JVXaL1hDuFBCrKjoTGS50b+IlCasXj3/2qxqLx4+4FyoOOi9u1Wb/Tph+DXxwG6T0htvnwpl3NKjyX7l1L5c9+w1PfJrLub1S+OLno5k4rKtV/sa0IBYAGuLICKAUeiS3Iyqiaf+M1YvHf7e5hAXrdx/dkNrPmZxt4QtQ7jOFtCqseBeeGw75X8JFv4effALJPet97rKKKh77ZA2XPfsNu/Yf5vlrB/Pi9UNJbW8rdBnT0lgAaAhPLrRNZLEnvEnb/31VLx5/3LKRI++D0mL47g3n+4Fd8Pfr4L2boEOWc9d/9t0NmkF0fkExY5+ezYtfr+eqwel8cd9oxg1o2ERwxpjgs8ngGsKzlqqkXhTmlzFhWNMMAa3phIvHdz0Tup4F8/4CbROdtv7yg3DBw3D2PQ2q+PcequDxT9bw9rdb6NohhrduHs7ZPW1MvzEtnT0B1JcqeNawJ7YHwNH1eoPghIvHj5gEe7fA+zc7Qzpvmw0jJzWo8v905Q4ufPJr/r5oC7ed04PPJp1jlb8xrYQ9AdTXwSI4tOfIHEBN9Q5AbU64eHz2Rc5CMR16wFn3QHj9/zPv2l/GQx+t4pOVO+jTuT1//fEZDEi3Mf3GtCYWAOrryCpgnYmODCOjQ0xQi/OTs7vz0uz1PP9VAU9ePchJDAtzZu5sAFXlncVb+P3Hayir9PJfP+jFref0sDH9xrRCFgDqq8gZAbTwQArZHeMID/Kwx8TYKK4Z3pXX5m3kvgtzTikgbSo+yAPvr2BeQTHDMjvw+JUD6JESnD4OY0zjs9u6+vLkQVQcC4vaNOkLYCdzy6gehAnHLx7vp8oqLy9+XcAPnp7NisK9/P6K/ky/5Uyr/I1p5ewJoL48uVQm5bBzQzm9OjWPCrJTfDRXDUl3Fo8fk03HOP/H5K/atpfJ761gxda9XNg3ld9d1p9O8Tam35hQYE8A9eXJY09MJhDcEUA11bp4/EmUVVTxxKe5XPrMN2zfe4hnrxnM1OuHWOVvTAixJ4D6OLQHDuxkc1rwRwDVdGTx+PmbuHO0z+LxtViwvpgH3l/BhqKD/GhIOv99cR8SYqKasLTGmObAngDqw11wZXV5F+LaRNC5md0t17p4vI99ZRX8+oMVTJi6gEqvlzduGs4ff3SaVf7GhCh7AqgPdwjoggMp5HSKo7mtYnnCxeOBz1ft4DcfrcSz/zC3jMrkvgtziImy//zGhDJ7AqgPTx4a0Zb5RW2bzQigmmouHu/Zf5i73lzKrdOWkBgTxQd3juC/L+5rlb8xxp4A6sWTS2WHnuzeXNVky0DW1+CuiZzZowMvzVlPuzYRPPZJLocqquyFLmPMcaw2qI+iteyJceYAak4dwDVVLx4/+f0V9EqN45N7R3HXeT2t8jfGHMOvGkFExopInojki8jkWrZ3FZFZIvKdiCwXkfFuencROSQi37s/L/gcM0REVrh5/kWaW4N6TYf3w94tbHbnAArWNND+GNkzmdtHZ/HoFQOYfuuZZNkLXcaYWtTZBOQu6v4scCFQCCwSkRnuMpDVHsRZK/h5EemLs3xkd3dbgaoOqiXr54FbgIXu/mOBTxp4HY2vyB0BVNGZpNgoktu1CXKBTkxEmDyud7CLYYxp5vx5AhgG5KvqelUtB6YDl9XYR4H27ud4YNvJMhSRzkB7VV3gLh7/N+Dy+hS8yXmOzgHUXDuAjTGmPvwJAGnAFp/vhW6arynAdSJSiHM3f4/Ptky3aehrERnlk2dhHXk2L548NCySOUXtmm0HsDHG1EegegUnAq+pajowHpgmImHAdqCrqp4O/Bx4S0TanySf44jIrSKyWEQWezyeABW3ATx5VCZmsa+8ebf/G2OMv/wJAFuBDJ/v6W6ar5uAdwBUdT4QDSSr6mFVLXbTlwAFQI57fHodeeIeN1VVh6rq0JSUFD+K20g8uex25wBqziOAjDHGX/4EgEVAtohkikgUMAGYUWOfzcAYABHpgxMAPCKS4nYiIyI9gGxgvapuB/aJyJnu6J8bgI8CckWNoeIQ7NnI5jAnDjanSeCMMaah6hwFpKqVInI38BkQDryiqqtE5BFgsarOAO4HXhKR+3A6hG9UVRWRc4BHRKQC8AK3q+puN+s7gdeAtjijf5rvCKDifEBZXdmZzvHRxLc98URrxhjTUvj1JrCqzsTp3PVN+63P59XAiFqOew947wR5Lgb616ewQVM9Amh/it39G2NaDXs11B+eXFTC+Xp3PL1sBJAxppWwAOAPTy4V8d04WBlu7wAYY1oNCwD+8PjMAWRDQI0xrYQFgLpUlsPuAjaFZSACPTtaE5AxpnWwAFCX3evBW8nqys5kJMbYPPrGmFbDAkBd3FXAFu63OYCMMa2LBYC6ePJQhDl7EunVyZp/jDGthwWAuhTlURGXwQFvlD0BGGNaFQsAdfHkHZ0DyEYAGWNaEQsAJ1NVCUXr2BKWQUSY0CPZmoCMMa2HBYCTKdkEVYdZWdmZ7smxREXYn8sY03pYjXYyPnMA2RTQxpjWxgLAybhDQL/Z28E6gI0xrY4FgJPx5FEe04n9GmNDQI0xrY4FgJPx5B6ZA8ieAIwxrY0FgBPxeqFoHZvCM4iKCKNbUmywS2SMMQFlAeBE9hVCxUFWVXSmZ0o7wsMk2CUyxpiA8isAiMhYEckTkXwRmVzL9q4iMktEvhOR5SIyvpbtB0TkFz5pG0VkhYh8LyKLT/1SAswdAbRgf4q9AGaMaZXqDADuou7PAuOAvsBEEelbY7cHgXdU9XScReOfq7H9SWpf8/c8VR2kqkPrXfLGdmQSuGRr/zfGtEr+PAEMA/JVdb2qlgPTgctq7KNAe/dzPLCteoOIXA5sAFadcmmbkiePiugkSoizEUDGmFbJnwCQBmzx+V7opvmaAlwnIoU4i8ffAyAi7YBfAQ/Xkq8Cn4vIEhG5tZ7lbnyePHa7I4CyO9oTgDGm9QlUJ/BE4DVVTQfGA9NEJAwnMDylqgdqOWakqg7GaVq6S0TOqS1jEblVRBaLyGKPxxOg4tZBFTx5bArLIDYqnLSEtk1zXmOMaUL+LG+1Fcjw+Z7upvm6CRgLoKrzRSQaSAaGA1eJyB+ABMArImWq+oyqbnX33yUiH+A0Nc2ueXJVnQpMBRg6dKjW49oabv8OOLyXVdGdyE6NI8xGABljWiF/ngAWAdkikikiUTidvDNq7LMZGAMgIn2AaMCjqqNUtbuqdgeeBh5V1WdEJFZE4tz9Y4GLgJWBuKCAKLI5gIwxrV+dAUBVK4G7gc+ANTijfVaJyCMicqm72/3ALSKyDHgbuFFVT3a3ngrMdff/FvhYVT89lQsJKHcI6JLSVHJsCKgxppXya4VzVZ2J07nrm/Zbn8+rgRF15DHF5/N64LT6FLRJeXKpiIrHUxZPTqqNADLGtE72JnBtPHnsbtsdEGsCMsa0WhYAauPJZVN4VxJiIkmJaxPs0hhjTKPwqwkopBwsgtJiVoV1Jic1DhEbAWSMaZ3sCaCmI3MAJVvzjzGmVbMAUJM7B9CKw52sA9gY06pZAKjJk0dlRCzbSLJJ4IwxrZoFgJqKjo4AsgBgjGnNLADU5M4B1DGuDYmxUcEujTHGNBoLAL4OlcD+7ayq6GSLwBhjWj0LAL6K1gLOHEDW/GOMae0sAPhyh4CuquxiI4CMMa2eBQBfnlyqwtpQqPYEYIxp/SwA+PLksbttN7yEkW0BwBjTylkA8OWOAEpPbEu7NjZLhjGmdbMAUK38IOzdzMryzjYFhDEmJFgAqOaOAFp0MMWaf4wxIcECQDV3BFBuVRd6dbIRQMaY1s+vACAiY0UkT0TyRWRyLdu7isgsEflORJaLyPhath8QkV/4m2eT8+TilQg2aaqNADLGhIQ6A4CIhAPPAuOAvsBEEelbY7cHcdYKPh1n0fjnamx/Eviknnk2Lc9adkdn4JUIslLsCcAY0/r58wQwDMhX1fWqWg5MBy6rsY8C7d3P8cC26g0icjmwAVhVzzyblieXTWEZdE+OJToyPKhFMcaYpuBPAEgDtvh8L3TTfE0BrhORQpzF4+8BEJF2wK+AhxuQZ9OpKIM9G1hZ3pmcjtb8Y4wJDYHqBJ4IvKaq6cB4YJqIhOEEhqdU9UBDMxaRW0VksYgs9ng8gSltTcX5oF6WlKaQY5PAGWNChD9vO20FMny+p7tpvm4CxgKo6nwRiQaSgeHAVSLyByAB8IpIGbDEjzxx85sKTAUYOnSo+lHe+nNXAVvrTecH1gFsjAkR/gSARUC2iGTiVNITgGtq7LMZGAO8JiJ9gGjAo6qjqncQkSnAAVV9RkQi/Miz6RStRQljg3ayIaDGmJBRZwBQ1UoRuRv4DAgHXlHVVSLyCLBYVWcA9wMvich9OB3CN6rqCe/WT5RnAK6nYTy57IlOQyui6ZYUG7RiGGNMU/JrwhtVnYnTueub9lufz6uBEXXkMaWuPIPGk8cmSadHSiyR4fZunDEmNFhtV1UBxfnOCCBr/zfGhBCb8nL3BvBWsrQ81ZaBNMaEFHsCcEcArdM0ewIwxoQUCwDuJHAF2sWmgTbGhBRrAvLkUhLVCapiSU9sG+zSGGNMk7EngKI8NoZlkJ3ajrAwCXZpjDGmyYR2APBWQdE6GwFkjAlJod0EVLIJKstYVtHJ2v+NMSEntJ8AqjuAvV1sEjhjTMixAADka5o9ARhjQk5oNwF58tgXmYwST2r7NsEujTHGNKkQfwLIZaNk0Cs1DhEbAWSMCS2hGwBU0aK1rCzvZO3/xpiQFLoBYN9WpPwAqyo6W/u/MSYkhW4AqJ4DyGtzABljQlMIBwBnBJAzCZytAmaMCT0hHAByORAeT3i7ZJLa2QggY0zo8SsAiMhYEckTkXwRmVzL9q4iMktEvhOR5SIy3k0fJiLfuz/LROQKn2M2isgKd9viwF2SnzzOHEDW/GOMCVV1vgcgIuHAs8CFQCGwSERmuMtAVnsQeEdVnxeRvjhLPXYHVgJD3TWAOwPLROQfqlrpHneeqhYF8Hr8o4p68lhZfoYFAGNMyPLnCWAYkK+q61W1HJgOXFZjHwXau5/jgW0AqlrqU9lHu/sF34FdSFkJayq72CpgxpiQ5U8ASAO2+HwvdNN8TQGuE5FCnLv/e6o3iMhwEVkFrABu9wkICnwuIktE5NYGlr9h3BFA+drFOoCNMSErUJ3AE4HXVDUdGA9ME5EwAFVdqKr9gDOAB0Qk2j1mpKoOBsYBd4nIObVlLCK3ishiEVns8XgCU9rqEUDedLKtCcgYE6L8CQBbgQyf7+lumq+bgHcAVHU+TnNPsu8OqroGOAD0d79vdX/vAj7AaWo6jqpOVdWhqjo0JSXFj+L6oSiPQ2GxRLTvRPvoyMDkaYwxLYw/AWARkC0imSISBUwAZtTYZzMwBkBE+uAEAI97TISb3g3oDWwUkVgRiXPTY4GLcDqMm4Ynjw2SQU7n9nXva4wxrVSdo4DcETx3A58B4cArqrpKRB4BFqvqDOB+4CURuQ+nbf9GVVURGQlMFpEKwAvcqapFItID+MCdgC0CeEtVP22UK6ztmjy5rKroZ1NAGGNCml/TQavqTJzOXd+03/p8Xg2MqOW4acC0WtLXA6fVt7ABUbobOeghryqN3hYAjGkxKioqKCwspKysLNhFabaio6NJT08nMtK/pu3QWw/AZxGYyywAGNNiFBYWEhcXR/fu3W369lqoKsXFxRQWFpKZmenXMaE3FcSRIaBp9OxoQ0CNaSnKyspISkqyyv8ERISkpKR6PSGFYADIo0yiiUzMoG1UeLBLY4ypB6v8T66+f58QDAC5bJI0sjvFB7skxhgTVCEXANSTx+oKmwLCGFN/O3bsYMKECWRlZTFkyBDGjx9PWFgYeXl5x+w3adIknnjiiSCV0n+hFQDK9iH7t7HWm2ZvABtj6kVVueKKKzj33HMpKChgyZIlPPbYY4wePZrp06cf2c/r9fLuu+8yYcKEIJbWP6E1CqhoLeAsAnO5BQBjWqyH/7GK1dv2BTTPvl3a89B/9Dvh9lmzZhEZGcntt99+JO20007jL3/5C1dffTUPPfQQALNnz6Zbt25069YtoOVrDKH1BOCOANpAGpnJsUEujDGmJVm5ciVDhgw5Ln3AgAGEhYWxbNkyAKZPn87EiRObungNElpPAJ48KiSSyORMoiJCK/YZ05qc7E49GCZOnMj06dPp168fH374IQ8//HCwi+SX0KoFPXlsIo2enRKCXRJjTAvTr18/lixZUuu2CRMm8M477/DFF18wcOBAUlNTm7h0DRNSAcDryWV1ZWdbBcwYU2/nn38+hw8fZurUqUfSli9fzpw5c8jKyiI5OZnJkye3mOYfCKUAUF6KlGwm35tmAcAYU28iwgcffMAXX3xBVlYW/fr144EHHqBTp06A0wyUm5vLlVdeGeSS+i90+gCK1yEo6zSNK+wdAGNMA3Tp0oV33nmn1m2TJk1i0qRJTVugUxQ6TwDuJHCbwjLo2iEmyIUxxpjgC6EAkEsl4UR1zCI8zOYTMcaYEAoAeWyhEz1SOwS7JMYY0yyETACo2rmG3Kou5Fj7vzHGAH4GABEZKyJ5IpIvIpNr2d5VRGaJyHcislxExrvpw0Tke/dnmYhc4W+eAVV5mLCSjazTNFsG0hhjXHUGABEJB54FxgF9gYki0rfGbg8C76jq6TiLxj/npq8EhqrqIGAs8KKIRPiZZ+AUFyBaRb433Z4AjDHG5c8TwDAgX1XXq2o5MB24rMY+CrR3P8cD2wBUtVRVK930aHc/f/MMHHcOoK2RXekSH91opzHGtG4bN26kf//+fu9/44038u677x6X/tVXX3HJJZc0qAwlJSU899xzde/oB38CQBqwxed7oZvmawpwnYgU4iwef0/1BhEZLiKrgBXA7W5A8CfPwPHk4UWI6JhtKwoZY1q0QAaAQL0INhF4TVX/JCJnAdNEpL+qelV1IdBPRPoAr4vIJ/XJWERuBW4F6Nq1a4MKp0V5FNKRzE7JDTreGNPMfDIZdqwIbJ6dBsC4x+vcraqqiltuuYV58+aRlpbGRx99RF5eHrfffjulpaVkZWXxyiuvkJiYeMxxn376KZMmTSImJoaRI0ceSd+9ezc//elPWb9+PTExMUydOpWBAwcyZcoU2rVrxy9+8QsA+vfvzz//+U8mT55MQUEBgwYN4sILL+SPf/xjgy/ZnyeArUCGz/d0N83XTcA7AKo6H6e555jaVlXXAAeA/n7mWX3cVFUdqqpDU1JS/Cju8ap25pJXZVNAGGNO3bp167jrrrtYtWoVCQkJvPfee9xwww088cQTLF++nAEDBhw3G2hZWRm33HIL//jHP1iyZAk7duw4su2hhx7i9NNPZ/ny5Tz66KPccMMNJz3/448/TlZWFt9///0pVf7g3xPAIiBbRDJxKukJwDU19tkMjAFec+/0owGPe8wWVa0UkW5Ab2AjUOJHngGTN+AX/HXbBu6xDmBjWgc/7tQbS2ZmJoMGDQJgyJAhFBQUUFJSwujRowH48Y9/zI9+9KNjjsnNzSUzM5Ps7GwArrvuuiOTys2dO5f33nsPcCacKy4uZt++wC52cyJ1BgC38r4b+AwIB15R1VUi8giwWFVnAPcDL4nIfTgdvTeqqorISGCyiFQAXuBOVS0CqC3PxrhAgIXhQ1jgbcv/2ROAMeYUtWnT5sjn8PBwSkpKGuU8EREReL3eI9/LysoCfg6/3gNQ1ZmqmqOqWar6ezftt27lj6quVtURqnqaqg5S1c/d9Gmq2s9NG6yqH54sz8aydud+EmMiSW4X1ZinMcaEoPj4eBITE5kzZw4A06ZNO/I0UK13795s3LiRgoICAN5+++0j20aNGsWbb74JOKODkpOTad++Pd27d2fp0qUALF26lA0bNgAQFxfH/v37A1L2kJgNNG/nfnJS42wEkDGmUbz++utHOoF79OjBq6++esz26Ohopk6dysUXX0xMTAyjRo06UolPmTKFn/70pwwcOJCYmBhef/11AH74wx/yt7/9jX79+jF8+HBycnIASEpKYsSIEfTv359x48adUj+AqGrdezUTQ4cO1cWLF9f7uN/9czWd46O5eVSPRiiVMaYprFmzhj59+gS7GM1ebX8nEVmiqkNr7hsSTwC/uaTxXjI2xpiWKmQmgzPGGHMsCwDGmBajJTVZB0N9/z4WAIwxLUJ0dDTFxcUWBE5AVSkuLiY62v/5zkKiD8AY0/Klp6dTWFiIx+MJdlGarejoaNLT0/3e3wKAMaZFiIyMJDMzM9jFaFWsCcgYY0KUBQBjjAlRFgCMMSZEtag3gUXEA2wKdjnqKRkoCnYhmphdc2iwa245uqnqcfPpt6gA0BKJyOLaXsFuzeyaQ4Ndc8tnTUDGGBOiLAAYY0yIsgDQ+KYGuwBBYNccGuyaWzjrAzDGmBBlTwDGGBOiLAAYY0yIsgDQSEQkQ0RmichqEVklIvcGu0xNRUTCReQ7EflnsMvSFEQkQUTeFZFcEVkjImcFu0yNSUTuc/+fXikib4uI/9NPtiAi8oqI7BKRlT5pHUTkXyKyzv2dGMwynioLAI2nErhfVfsCZwJ3iUioLE12L7Am2IVoQn8GPlXV3sBptOJrF5E04GfAUFXtD4QDE4JbqkbzGjC2Rtpk4EtVzQa+dL+3WBYAGomqblfVpe7n/TiVQlpwS9X4RCQduBh4OdhlaQoiEg+cA/wVQFXLVbUkqIVqfBFAWxGJAGKAbUEuT6NQ1dnA7hrJlwGvu59fBy5vyjIFmgWAJiAi3YHTgYVBLkpTeBr4JeANcjmaSibgAV51m71eFpHYYBeqsajqVuB/gc3AdmCvqn4e3FI1qVRV3e5+3gGkBrMwp8oCQCMTkXbAe8AkVd0X7PI0JhG5BNilqkuCXZYmFAEMBp5X1dOBg7TwZoGTcdu8L8MJfF2AWBG5LrilCg51xtC36HH0FgAakYhE4lT+b6rq+8EuTxMYAVwqIhuB6cD5IvJGcIvU6AqBQlWtfrp7FycgtFYXABtU1aOqFcD7wNlBLlNT2ikinQHc37uCXJ5TYgGgkYiI4LQLr1HVJ4Ndnqagqg+oarqqdsfpGPy3qrbqu0NV3QFsEZFebtIYYHUQi9TYNgNnikiM+//4GFpxp3ctZgA/dj//GPgoiGU5ZRYAGs8I4Hqcu+Dv3Z/xwS6UaRT3AG+KyHJgEPBocIvTeNwnnXeBpcAKnDqkVU2PUE1E3gbmA71EpFBEbgIeBy4UkXU4T0OPB7OMp8qmgjDGmBBlTwDGGBOiLAAYY0yIsgBgjDEhygKAMcaEKAsAxhgToiwAGGNMiLIAYIwxIer/A8iWtoquWdNSAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "from sklearn.model_selection import cross_val_score\n",
    "\n",
    "cv_scores, holdout_scores = [], []\n",
    "n_neighb = [1, 2, 3, 5, 7, 9, 11]\n",
    "\n",
    "for k in n_neighb:\n",
    "\n",
    "    knn = KNeighborsClassifier(n_neighbors=k)\n",
    "    cv_scores.append(np.mean(cross_val_score(knn, X_train, y_train, cv=5)))\n",
    "    knn.fit(X_train, y_train)\n",
    "    holdout_scores.append(accuracy_score(y_holdout, knn.predict(X_holdout)))\n",
    "\n",
    "plt.plot(n_neighb, cv_scores, label='CV')\n",
    "plt.plot(n_neighb, holdout_scores, label='holdout')\n",
    "plt.title('Easy task. kNN fails')\n",
    "plt.legend();"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Рассмотрим еще один набор данных."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 128,
   "metadata": {},
   "outputs": [],
   "source": [
    "path1 = '../../Dataset/cardio.csv'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 129,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(path1, index_col='id', sep=';')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 130,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(70000, 12)"
      ]
     },
     "execution_count": 130,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 131,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>age</th>\n",
       "      <th>gender</th>\n",
       "      <th>height</th>\n",
       "      <th>weight</th>\n",
       "      <th>ap_hi</th>\n",
       "      <th>ap_lo</th>\n",
       "      <th>cholesterol</th>\n",
       "      <th>gluc</th>\n",
       "      <th>smoke</th>\n",
       "      <th>alco</th>\n",
       "      <th>active</th>\n",
       "      <th>cardio</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>id</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>18393</td>\n",
       "      <td>2</td>\n",
       "      <td>168</td>\n",
       "      <td>62.0</td>\n",
       "      <td>110</td>\n",
       "      <td>80</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>20228</td>\n",
       "      <td>1</td>\n",
       "      <td>156</td>\n",
       "      <td>85.0</td>\n",
       "      <td>140</td>\n",
       "      <td>90</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>18857</td>\n",
       "      <td>1</td>\n",
       "      <td>165</td>\n",
       "      <td>64.0</td>\n",
       "      <td>130</td>\n",
       "      <td>70</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>17623</td>\n",
       "      <td>2</td>\n",
       "      <td>169</td>\n",
       "      <td>82.0</td>\n",
       "      <td>150</td>\n",
       "      <td>100</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>17474</td>\n",
       "      <td>1</td>\n",
       "      <td>156</td>\n",
       "      <td>56.0</td>\n",
       "      <td>100</td>\n",
       "      <td>60</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      age  gender  height  weight  ap_hi  ap_lo  cholesterol  gluc  smoke  \\\n",
       "id                                                                          \n",
       "0   18393       2     168    62.0    110     80            1     1      0   \n",
       "1   20228       1     156    85.0    140     90            3     1      0   \n",
       "2   18857       1     165    64.0    130     70            3     1      0   \n",
       "3   17623       2     169    82.0    150    100            1     1      0   \n",
       "4   17474       1     156    56.0    100     60            1     1      0   \n",
       "\n",
       "    alco  active  cardio  \n",
       "id                        \n",
       "0      0       1       0  \n",
       "1      0       1       1  \n",
       "2      0       0       1  \n",
       "3      0       1       1  \n",
       "4      0       0       0  "
      ]
     },
     "execution_count": 131,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "В данных присутствует 3 типа признаков:\n",
    "\n",
    "- *Objective*: фактическая информация;\n",
    "- *Examination*: результаты медицинского обследования;\n",
    "- *Subjective*: информация представленная пациентом.\n",
    "\n",
    "| Feature | Variable Type | Variable      | Value Type |\n",
    "|---------|--------------|---------------|------------|\n",
    "| Age | Objective Feature | age | int (days) |\n",
    "| Height | Objective Feature | height | int (cm) |\n",
    "| Weight | Objective Feature | weight | float (kg) |\n",
    "| Gender | Objective Feature | gender | categorical code |\n",
    "| Systolic blood pressure | Examination Feature | ap_hi | int |\n",
    "| Diastolic blood pressure | Examination Feature | ap_lo | int |\n",
    "| Cholesterol | Examination Feature | cholesterol | 1: normal, 2: above normal, 3: well above normal |\n",
    "| Glucose | Examination Feature | gluc | 1: normal, 2: above normal, 3: well above normal |\n",
    "| Smoking | Subjective Feature | smoke | binary |\n",
    "| Alcohol intake | Subjective Feature | alco | binary |\n",
    "| Physical activity | Subjective Feature | active | binary |\n",
    "| Presence or absence of cardiovascular disease | Target Variable | cardio | binary |\n",
    "\n",
    "Возраст представлен в днях от рождения.\n",
    "\n",
    "Целевой признак - cardio, которые означает, что человек в группе риска сердечных заболеваний.\n",
    "\n",
    "Выделим целевой признак и поделим набор данных в соотношении 70:30."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 132,
   "metadata": {},
   "outputs": [],
   "source": [
    "target = df['cardio']\n",
    "df.drop('cardio', axis = 1, inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 133,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train, X_holdout, y_train, y_holdout = train_test_split(df, target, test_size=0.3,\n",
    "                                                          random_state=17)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Задание в стиле \"free ride\"\n",
    "\n",
    "Используя полученные наборы данных для train и test необходимо:\n",
    "- получить максимальные значения метрик accuracy и recall на тестовой выборке с использованием дерева решений;\n",
    "- получить максимальные значения метрик accuracy и recall на тестовой выборке с использованием метода ближайших соседей;\n",
    "\n",
    "Попробуйте добавить новые признаки в модель для улучшения показателя метрик, например:\n",
    "- возраст в годах - поделить признак возраст на 365.25 ;create \"age in years\" (full age) - the remainder of dividing age in days by 365.25\n",
    "- создать 3 бинарных признака, основанных на признаке cholesterol используя get_dummies или oneHotEncoder (изучить самостоятельно);\n",
    "-  создать 3 бинарных признака основанных на признаке gluc;\n",
    "- и т.д."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}