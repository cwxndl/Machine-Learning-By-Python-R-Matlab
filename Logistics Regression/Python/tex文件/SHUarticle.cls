%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% github:https://github.com/cwxndl/Latex-for-SHU
% 作者: cwxndl
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{SHUarticle}[2023/04/29 version 1.0]
% 设置纸张为A4纸,行距为1.5倍
\LoadClass[oneside,a4paper,UTF8,linespread=1.5]{ctexart}
\author{ 虹 \thanks{微信公众号：数学与智能科学} }
\def\keyword#1{\gdef\@keyword{#1}}

%设置中英文摘要
\def\keyword#1{\gdef\@keyword{#1}}
\newcommand{\enabstractname}{Abstract}
\newcommand{\cnabstractname}{\heiti{本文概要}} %你可以将“本文概要”修改成你自己的命名，比如：摘要、概要
\newenvironment{enabstract}{%
	\par\small
	\noindent\mbox{}\hfill{\bfseries\enabstractname}\hfill\mbox{}\par
	\vskip 2.5ex}{\par\vskip2.5ex}
\newenvironment{cnabstract}{%
	\par\normalsize
	\noindent\mbox{}\hfill{\bfseries\cnabstractname}\hfill\mbox{}\par
	\vskip 2.5ex}
{\phantom{1}\\关键词：\@keyword\thispagestyle{empty}\setcounter{page}{0}\newpage}



% 设置附录格式
\ctexset{
	appendix={
		name={附录,：},
		number=\arabic{section},
	},
section ={
format ={\large\bfseries\heiti}
},
subsection = {
format = {\heiti}
},
subsubsection = {
format = {\heiti}
}
}
\setCJKmainfont[Mapping=fullwidth-stop]{SimSun}
\setmainfont{Times New Roman}

\renewcommand*{\textfraction}{0.05}
\renewcommand*{\topfraction}{0.9}
\renewcommand*{\bottomfraction}{0.8}
\renewcommand*{\floatpagefraction}{0.85}

\RequirePackage[top=1.5cm,bottom=1.5cm,left=2cm,right=2cm]{geometry} %页边距
\RequirePackage{titletoc}
\titlecontents{section}[0pt]{\addvspace{2pt}\filright}%
{\heiti\contentspush{\thecontentslabel\ }}%
{}{\titlerule*[8pt]{.}\contentspage}
\RequirePackage{fancyhdr,lastpage}
\RequirePackage{amsthm,amsmath,amsfonts,amssymb}
\let\lvert\relax\let\rvert\relax\let\lVert\relax\let\rVert\relax
\RequirePackage{newtxmath}
\RequirePackage{esint}
\DeclareSymbolFont{CMlargesymbols}{OMX}{cmex}{m}{n}
\let\sum\relax\let\prod\relax
\DeclareMathSymbol{\sum}{\mathop}{CMlargesymbols}{"50}
\DeclareMathSymbol{\prod}{\mathop}{CMlargesymbols}{"51}
\newtheorem{theorem}{定理}
\newtheorem{lemma}{引理} %引理
\newtheorem{proposition}{命题}
% 加载物理、国标宏包
\RequirePackage{physics,siunitx}

% 设置虚数,自然底数,微分符号
\def\ee{\mathrm{e}}
\def\ii{\mathrm{i}}
\def\leq{\leqslant}
\def\geq{\geqslant}

% 载入表格宏包和插图宏包
\RequirePackage{booktabs,tabularx,multirow,longtable,makecell,array}
\newcolumntype{Y}{>{\centering\arraybackslash}X}
\RequirePackage{graphicx}

% 设置图片路径
%\graphicspath{{..Images/}} % 如果你的图片目录在tex文件的父文件的同一级别Images里
\graphicspath{{figures/}}
\RequirePackage{caption}
\RequirePackage[section]{placeins}

%% 设置页眉logo
\RequirePackage{fancyhdr,lastpage}
\renewcommand\headrulewidth{0pt}
\fancyfoot[C]{\thepage{}}
\pagestyle{fancy}
%\fancyhf{}
\lhead{}  %在此处插入logo.pdf图片 图片靠左
\chead{} % 页眉中间位置内容
\rhead{\includegraphics[scale=0.13]{figures/shu.png}} %页眉右边位置内容

\captionsetup{format=hang}% 设置浮动体标题悬挂缩进
\DeclareCaptionLabelSeparator*{sspace}{\ \ }
\captionsetup[figure]{labelsep=sspace}
\captionsetup[table]{labelsep=sspace}
\DeclareCaptionFont{heiti}{\heiti}
\DeclareCaptionFont{5hao}{\zihao{5}}
\captionsetup{labelfont={heiti,5hao},textfont={heiti,5hao}}

% 载入颜色宏包
\RequirePackage{color,xcolor}

% 载入参考文献宏包，设置引用格式
\RequirePackage{natbib}
\setcitestyle{square,super}

% 载入代码宏包
\RequirePackage{listings}
% 设置代码的默认样式
\lstset{
	frame=none,% 取消边框
	breaklines=true,% 允许自动断行
	% breakatwhitespace=true,% 使用此命令仅允许在空格处自动断行
	showstringspaces=false,% 不显示字符串中的空格
	basicstyle=\small\ttfamily,% 设置代码基本样式
	flexiblecolumns=true,% 改善字母间距
	keywordstyle=\color{blue},% 设置关键词样式
	stringstyle=\color[rgb]{0.75,0,0.75},% 设置字符串样式
	commentstyle=\songti\color[rgb]{0,0.5,0},% 设置注释样式
	tabsize=4,% 设置制表符缩进
}

% 设置matlab代码环境
\lstnewenvironment{matlab}[1][]{
	\lstset{
		language=Matlab,
		% deletekeywords={disp},% 可在此行去除特定关键词的语法高亮		
		keywordstyle=\color{blue},% 设置关键词样式
		stringstyle=\color[rgb]{0.75,0,0.75},% 设置字符串样式
		commentstyle=\songti\color[rgb]{0,0.5,0},% 设置注释样式
		#1
	}
}{}

% 设置python代码环境
\lstnewenvironment{python}[1][]{
	\lstset{
		language=Python,
		keywordstyle=\color[RGB]{255,119,0},% 设置Keywords样式
		morekeywords={as},% 将特定单词加入Kewords中
		deletekeywords={print},%从 keywords中去除特定单词
		keywordstyle=[2]\color[RGB]{144,0,144},% 设置Builtins样式
		morekeywords=[2]{print},% 将特定单词加入Builtins中
		stringstyle=\color[RGB]{0,170,0},% 设置字符串样式
		commentstyle=\songti\color[RGB]{221,0,0},% 设置注释样式	
		#1
	}
}{}



% 载入超链接宏包
\RequirePackage{hyperref}
\hypersetup{breaklinks,colorlinks}% 允许超链接断行,取消超链接的边框,使链接文字带颜色
\hypersetup{hidelinks,bookmarksnumbered=true,bookmarksopen=true,pdfstartview=Fit}% 取消超链接的颜色，书签目录增加编号
%  设置水印
%%%%%%%%%%%%%%%%%%%%%%%
\RequirePackage{draftwatermark}         % 所有页加水印
\SetWatermarkText{数学与智能科学}           % 设置水印内容
% \SetWatermarkText{\includegraphics{fig/ZJDX-WaterMark.eps}}         % 设置水印logo
\SetWatermarkLightness{0.9}             % 设置水印透明度 0-1
\SetWatermarkScale{0.4} 
\RequirePackage{cleveref}% 载入自动引用宏包
\crefname{figure}{图}{图}
\crefname{table}{表}{表}
\crefname{equation}{式}{式}
\renewcommand{\eqref}[1]{\labelcref{#1}~\namecref{#1}}

%% 设置算法输出
\RequirePackage{algorithm}
\RequirePackage{algorithmic}
\renewcommand{\algorithmicrequire}{ \textbf{Input:}}     % use Input in the format of Algorithm  
\renewcommand{\algorithmicensure}{ \textbf{Initialize:}} % use Initialize in the format of Algorithm  
\renewcommand{\algorithmicreturn}{ \textbf{Output:}}     % use Output in the format of Algorithm  
