"""50 도형·설명·정렬 점수·연속 prefix·조건 교환을 표시합니다."""
import numpy as np
from . import concept_plots as cp
from .gradient_plots import _clean_axes


def plot_image_text_pairs(images,captions):
    """본문에서 만든 RGB 이미지와 대응 설명을 그립니다."""
    fig,axes=cp.flow([2.4]*len(images),height=2.6,wspace=.35)
    for ax,image,caption in zip(axes,images,captions):
        ax.imshow(image)
        ax.set_title(caption,color=cp.INK,fontsize=11)
        ax.set_axis_off()
    cp.show(fig)


def plot_alignment_scores(image_vectors,text_vectors,scores):
    """이미지·텍스트 표현과 내적 점수의 축을 표시합니다."""
    fig,axes=cp.flow([3,3,3],height=2.8,wspace=.4)
    for ax,value,title in zip(axes,[image_vectors,text_vectors,scores],['image 표현 (N,K)','text 표현 (N,K)','image × text 점수']):
        cp.draw_grid(ax,value,title,kind='signed',fmt='{:.2f}')
    cp.show(fig)


def plot_positive_pairs(unique,repeated):
    """한 배치의 고유 설명과 반복 설명의 의미상 양성 관계를 표시합니다."""
    fig,axes=cp.flow([3.5,3.5],height=3,wspace=.4)
    cp.draw_grid(axes[0],unique,'범주당 한 쌍: 대각선 정답',kind='plain')
    cp.draw_grid(axes[1],repeated,'설명 반복: 여러 의미상 양성',kind='plain')
    cp.show(fig)


def plot_multimodal_learning(history,ylabel,title):
    """전달된 학습 진단을 표시합니다."""
    fig=cp.new_figure((7,3.3))
    ax=fig.subplots()
    _clean_axes(ax,'갱신 횟수',ylabel,title)
    ax.plot(history[:,0],history[:,1],color=cp.BLUE)
    fig.subplots_adjust(left=.15,bottom=.2)
    cp.show(fig)


def plot_continuous_prefix(feature,bridge,prefix):
    """연속 image feature가 decoder 폭의 prefix로 바뀌는 곱을 표시합니다."""
    fig,axes=cp.flow([3,3,3],height=2.8,wspace=.4)
    for ax,value,title in zip(axes,[feature,bridge,prefix],['image feature (N,K)','bridge (K,D)','연속 prefix (N,D)']):
        cp.draw_grid(ax,value,title,kind='signed',fmt='{:.2f}')
    cp.show(fig)


def plot_condition_swap(images,original_outputs,swapped_outputs):
    """원래 이미지와 서로 다른 조건에서 실제 생성한 설명을 나란히 둡니다."""
    fig,axes=cp.flow([2.5]*len(images),height=3,wspace=.4)
    for ax,image,own,swapped in zip(axes,images,original_outputs,swapped_outputs):
        ax.imshow(image)
        ax.set_title('자기 조건: '+own+'\n교환 조건: '+swapped,fontsize=10,color=cp.INK)
        ax.set_axis_off()
    cp.show(fig)
