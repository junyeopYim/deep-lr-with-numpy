"""19번 ViT의 그림 배치. 패치·점수·attention·학습 수치는 노트북이 계산해 전달합니다."""

import numpy as np
from matplotlib.patches import Rectangle
from . import concept_plots as cp


def _axes(ax, xlabel, ylabel, title):
    """실험에 필요한 축만 남깁니다."""
    ax.spines[['top','right']].set_visible(False)
    ax.grid(False)
    ax.set_xlabel(xlabel,fontsize=9,color=cp.INK)
    ax.set_ylabel(ylabel,fontsize=9,color=cp.INK)
    ax.set_title(title,fontsize=10.5,color=cp.INK)
    ax.tick_params(labelsize=8)


def plot_patch_flow(image, patches, restored, patch_size):
    """작은 정수판의 패치 순서와 정확한 역변환을 나란히 보여 줍니다."""
    fig,axes=cp.flow([3.,3.,3.],height=3.2,wspace=.45)
    cp.draw_grid(axes[0],image,'4×4 원래 숫자판',kind='plain',fmt='{:.0f}')
    axes[0].add_patch(Rectangle((-.5,-.5),patch_size,patch_size,fill=False,ec=cp.GOLD,lw=2))
    cp.draw_grid(axes[1],patches,'행 하나 = 패치 하나',kind='plain',fmt='{:.0f}',row_labels=['좌상','우상','좌하','우하'])
    cp.draw_grid(axes[2],restored,'역변환한 숫자판',kind='plain',fmt='{:.0f}')
    cp.connect(fig,axes[0],axes[1],'순서 변경')
    cp.connect(fig,axes[1],axes[2],'되돌리기')
    cp.show(fig)


def plot_token_embedding(image, patches, base, position, tokens, patch_size):
    """내용 임베딩·CLS와 위치 표가 같은 모양에서 더해짐을 보여 줍니다."""
    fig,axes=cp.flow([2.2,2.4,1.8,1.8,1.8],height=3.5,wspace=.45)
    cp.draw_image(axes[0],image,'고정 숫자 7')
    axes[0].add_patch(Rectangle((-.5,-.5),patch_size,patch_size,fill=False,ec=cp.GOLD,lw=2))
    cp.draw_rows(axes[1],patches,'16패치 × 49화소')
    cp.draw_rows(axes[2],base,'CLS + 내용',kind='signed')
    cp.draw_rows(axes[3],position,'슬롯별 위치 R',kind='signed')
    cp.draw_rows(axes[4],tokens,'합한 17토큰',kind='signed')
    for ax in axes[2:]:
        cp.label(ax,'첫 행은 CLS 슬롯',y=-.11)
    cp.connect(fig,axes[1],axes[2],'49 → 6')
    cp.connect(fig,axes[2],axes[3],'+')
    cp.connect(fig,axes[3],axes[4],'=')
    cp.show(fig)


def plot_attention_block(tokens, normalized, attention, branch, output):
    """토큰별 LN, 행 attention, 수정량과 잔차 합을 직접 계산한 배열로 표시합니다."""
    fig,axes=cp.flow([2.1,2.1,2.4,2.1,2.1],height=2.8,wspace=.5)
    for ax,value,title in zip(axes,[tokens,normalized,attention,branch,output],
                             ['입력 토큰 Z','토큰별 LN','행 attention A','수정량 (AV)Wₒ','잔차 합 H']):
        cp.draw_grid(ax,value,title,fmt='{:.2f}',fontsize=9)
    cp.connect(fig,axes[0],axes[1],'LN')
    cp.connect(fig,axes[1],axes[2],'Q,K')
    cp.connect(fig,axes[2],axes[3],'V,Wₒ')
    cp.connect(fig,axes[3],axes[4],'+ Z')
    cp.label(axes[2],'행=query · 열=key',y=-.17)
    cp.show(fig)


def plot_permutation(original, shuffled, no_position_scores, position_scores):
    """내용 패치 순열 뒤 위치 유무별 고정 모델 점수를 나란히 보여 줍니다."""
    fig,axes=cp.flow([2.,2.,2.8,2.8],height=3.,wspace=.5)
    cp.draw_image(axes[0],original,'원래 조각 배치')
    cp.draw_image(axes[1],shuffled,'조각 순서만 변경')
    cp.draw_grid(axes[2],no_position_scores,'위치 없음 · CLS 점수',fmt='{:.3f}',row_labels=['원래','순열'])
    cp.draw_grid(axes[3],position_scores,'고정 위치 · CLS 점수',fmt='{:.3f}',row_labels=['원래','순열'])
    cp.connect(fig,axes[0],axes[1],'패치 순열')
    cp.label(axes[2],'같은 점수',y=-.2)
    cp.label(axes[3],'이 예에서는 달라진 점수',y=-.2)
    cp.show(fig)


def plot_patch_budget(patch_sizes, token_counts, parameter_counts, attention_entries):
    """본문에서 세어 온 구조별 원소 수를 표로 표시합니다."""
    fig,axes=cp.flow([6.7,3.],height=3.,wspace=.4)
    table=np.column_stack([patch_sizes,token_counts,parameter_counts,attention_entries])
    cp.draw_grid(axes[0],table,'같은 폭 d=12 · MLP 폭 24',kind='plain',fmt='{:.0f}',
                 col_labels=['패치 P','CLS 포함 L','파라미터','attention L²'])
    axes[0].set_aspect('auto')
    cp.draw_formula(axes[1],'$L=T+1$', '단일 head·표본 하나의 값\nL²는 전체 FLOPs가 아닙니다')
    cp.show(fig)


def plot_classifier_flow(image, patches, encoded, pooled, probabilities):
    """고정 이미지의 패치 목록·encoder 출력·CLS 읽기·초기 분류 확률입니다."""
    fig,axes=cp.flow([2.,2.2,2.1,1.4,2.7],height=3.4,wspace=.45)
    cp.draw_image(axes[0],image,'28×28 숫자 7')
    cp.draw_rows(axes[1],patches,'16×49 패치')
    cp.draw_rows(axes[2],encoded,'17×12 최종 LN',kind='signed')
    cp.draw_strip(axes[3],pooled,'CLS 12개 값',kind='signed')
    cp.draw_bars(axes[4],probabilities,'학습 전 클래스 확률',labels=[str(i) for i in range(10)],marker=7,xlim=(0,1))
    cp.connect(fig,axes[1],axes[2],'한 블록')
    cp.connect(fig,axes[2],axes[3],'첫 행')
    cp.connect(fig,axes[3],axes[4],'분류기')
    cp.label(axes[4],'금색 테두리는 정답 7',y=-.12)
    cp.show(fig)


def plot_vit_training(histories, names):
    """각 읽기 방식의 한 실행을 보여 주며 추정하지 않은 오차 막대는 넣지 않습니다."""
    fig=cp.new_figure((11.,3.6))
    axes=fig.subplots(1,2,gridspec_kw={'wspace':.3})
    _axes(axes[0],'epoch','표본 평균 CE','학습과 validation의 시점을 구분합니다')
    _axes(axes[1],'epoch','정확도','validation · 마지막 epoch로 평가')
    for history,name,color in zip(histories,names,[cp.BLUE,cp.RED]):
        axes[0].plot(history[:,0],history[:,1],ls='--',color=color,label=f'{name} train')
        axes[0].plot(history[:,0],history[:,2],color=color,label=f'{name} validation')
        axes[1].plot(history[:,0],history[:,3],marker='o',color=color,label=name)
    for ax in axes:
        ax.legend(frameon=False,fontsize=8)
        ax.set_xticks(histories[0,:,0])
    cp.show(fig)


def plot_vit_predictions(images, labels, predictions, names):
    """test 앞 10장을 빠짐없이 보여 주며 두 모델의 예측을 함께 적습니다."""
    fig,axes=cp.flow([2.]*5,rows=2,height=2.2,wspace=.25,hspace=.65)
    flat_axes=[ax for row in axes for ax in row]
    for i,(ax,image,label) in enumerate(zip(flat_axes,images,labels)):
        cp.draw_image(ax,image,f'정답 {label}')
        text=' · '.join(f'{name} {predictions[j,i]}' for j,name in enumerate(names))
        cp.label(ax,text,y=-.13)
    fig.suptitle('미리 정한 test 앞 10장 · 전체 지표는 600장',y=1.01,fontsize=11,color=cp.INK)
    cp.show(fig)


def plot_cls_attention(image, patch_probabilities, self_probability, prediction):
    """CLS가 자신과 패치에 준 원래 확률을 재정규화 없이 보여 줍니다."""
    fig,axes=cp.flow([2.5,3.8,3.3],height=3.2,wspace=.5)
    cp.draw_image(axes[0],image,'고정 개념 예시 · 숫자 7')
    cp.draw_grid(axes[1],patch_probabilities,'CLS → 패치 4×4',kind='plain',fmt='{:.3f}')
    cp.draw_formula(axes[2],f'예측 숫자 {prediction}',f'CLS 자신: {self_probability:.3f}\n패치 확률은 재정규화하지 않음\n인과적 중요도 지도가 아님')
    cp.show(fig)
