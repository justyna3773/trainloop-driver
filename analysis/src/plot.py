import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
from utils import Utils
import numpy as np
import pandas as pd
from captum.attr import visualization as viz
from viz import visualize_image_attr_multiple


POSITIVE_VALUE_COLOR = 'seagreen'
POSITIVE_VALUE_COLOR = 'mediumseagreen'
NEGATIVE_VALUE_COLOR = 'firebrick'
NEGATIVE_VALUE_COLOR = 'indianred'

def plot_mean_attributions_per_action(mean_attributions, action_names=Utils.ACTION_NAMES, feature_names=Utils.FEATURE_NAMES, policy='mlp', abs=False):
    if policy == 'mlp':
        for i, ig_attr in enumerate(mean_attributions):
            if abs:
                plt.figure(figsize=(10, 1.5))
                plt.title(f'Absolute mean attributions - Feature Importance for action: {action_names[i]}')
                mean_attributions_ = np.abs(ig_attr)
                df = pd.DataFrame(mean_attributions_, columns=feature_names)
                mean_attributions_ = df.mean(axis=0).sort_values(ascending=False)
                # ax[i].set_xticks(range(len(mean_attributions_.index)), mean_attributions_.index, rotation=30)
                plt.bar(x=[x for x in range(len(mean_attributions_.index))], height=mean_attributions_)
                plt.xticks(range(len(mean_attributions_.index)), mean_attributions_.index, rotation=30)

            else:
                x_axis_data = list(range(len(feature_names)))
                fig, ax = plt.subplots(nrows=len(mean_attributions), ncols=1, sharex=True, figsize=(20, 14))
                # plt.figure(figsize=(10, 2))
                # ax[0].title(f'Mean attributions for action: {action_names[i]}')
                ax[i].set_title(f'Mean attributions for action: {action_names[i]}')
                plt.xticks(x_axis_data, feature_names, rotation=30)
                # ax[i].set_ylim([-1, 1])
                color = []
                for val in ig_attr[0]:
                    if val < 0.0:
                        color.append(NEGATIVE_VALUE_COLOR)
                    else:
                        color.append(POSITIVE_VALUE_COLOR)
                ax[i].bar(x_axis_data, ig_attr[0], color=color)
    else:
        # attr_ig, delta = ig.attribute(img, n_steps=100, target=target, baselines=img * 0, return_convergence_delta=True)

        # attr_ig, delta = attribute_image_features(ig, img, baselines=img * 0, return_convergence_delta=True)
        # attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (0, 1))

        frames = []
        for i, ig_attr in enumerate(mean_attributions):

            print(f'Mean attributions for action: {action_names[i]}')
            # img_ = np.transpose(img.cpu().detach().numpy()[0], (1, 2, 0))
            attr_ig_ = np.transpose(ig_attr, (1, 2, 0))

            if attr_ig_.sum() == 0:
                print('Sum of attributions equals 0')
                continue

            fig_size = (20, 5)
            if abs:
                _ = visualize_image_attr_multiple(attr=attr_ig_,
                                                    original_image=attr_ig_, # TODO
                                                    methods=["blended_heat_map"],
                                                    signs=["absolute_value"],
                                                    show_colorbar=True,
                                                    titles=['Absolute mean attributions'],
                                                    fig_size=fig_size,
                                                    use_pyplot=True,
                                                    )
            else:
                _ = visualize_image_attr_multiple(attr=attr_ig_,
                                                    original_image=attr_ig_, # TODO
                                                    methods=["blended_heat_map", "blended_heat_map"],
                                                    signs=["positive", "negative"],
                                                    show_colorbar=True,
                                                    titles=['IG positive attributions', 'IG negative attributions'],
                                                    fig_size=fig_size,
                                                    use_pyplot=True,
                                                    )
            plt.show()

def plot_mean_attributions(mean_attributions, action_names=Utils.ACTION_NAMES, feature_names=Utils.FEATURE_NAMES, policy='mlp', abs=False):
    if policy == 'mlp':
        if abs:
            plt.figure(figsize=(10, 1.5))
            plt.title(f'Absolute mean attributions - Feature Importance')
            mean_attributions = np.abs(mean_attributions)
            df = pd.DataFrame(mean_attributions, columns=feature_names)
            mean_attributions = df.mean(axis=0).sort_values(ascending=False)
            plt.bar(x=[x for x in range(len(mean_attributions.index))], height=mean_attributions)
            plt.xticks(range(len(mean_attributions.index)), mean_attributions.index, rotation=30)

        else:
            x_axis_data = list(range(len(feature_names)))
            plt.figure(figsize=(10, 1.5))
            plt.title(f'Mean attributions')
            plt.xticks(x_axis_data, feature_names, rotation=30)
            # plt.ylim([-1, 1])
            color = []
            for val in mean_attributions[0]:
                if val < 0.0:
                    color.append(NEGATIVE_VALUE_COLOR)
                else:
                    color.append(POSITIVE_VALUE_COLOR)
            plt.bar(x_axis_data, mean_attributions[0], color=color)

    
    else:
        # attr_ig, delta = ig.attribute(img, n_steps=100, target=target, baselines=img * 0, return_convergence_delta=True)

        # attr_ig, delta = attribute_image_features(ig, img, baselines=img * 0, return_convergence_delta=True)
        # attr_ig = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (0, 1))

        frames = []
        # img_ = np.transpose(img.cpu().detach().numpy()[0], (1, 2, 0))
        attr_ig_ = np.transpose(mean_attributions, (1, 2, 0))

        if attr_ig_.sum() == 0:
            print('Sum of attributions equals 0')

        if abs:
            fig_size = (11, 2.5)
            _ = visualize_image_attr_multiple(attr=attr_ig_,
                                                original_image=attr_ig_, # TODO
                                                methods=["blended_heat_map"],
                                                signs=["absolute_value"],
                                                show_colorbar=True,
                                                titles=['Absolute mean attributions'],
                                                fig_size=fig_size,
                                                use_pyplot=True,
                                            )
        else:
            fig_size = (20, 5)
            _ = visualize_image_attr_multiple(attr=attr_ig_,
                                                original_image=attr_ig_, # TODO
                                                methods=["blended_heat_map", "blended_heat_map"],
                                                signs=["positive", "negative"],
                                                show_colorbar=True,
                                                titles=['Mean positive attributions', 'Mean negative attributions'],
                                                fig_size=fig_size,
                                                use_pyplot=True,
                                            )
        plt.show()

def render_env(img, feature_names=Utils.FEATURE_NAMES):
    plt.figure(figsize=(16, 5))
    color = []
    for val in img:
        if val < 0.0:
            color.append(NEGATIVE_VALUE_COLOR)
        else:
            color.append(POSITIVE_VALUE_COLOR)
    plt.xticks(range(7), feature_names, rotation=30)
    plt.bar(range(7), img, color=color)
    plt.title('Environment state')
    plt.show()

def plot_attributions(idx, attributions, img, action_names=Utils.ACTION_NAMES, feature_names=Utils.FEATURE_NAMES):
    fig, axs = plt.subplots(len(attributions)+1, 1, figsize=(10, 16), sharex=True)
    color = []
    for val in img:
        if val < 0.0:
            color.append(NEGATIVE_VALUE_COLOR)
        else:
            color.append(POSITIVE_VALUE_COLOR)
    # plt.xticks(range(7), feature_names, rotation=30)
    axs[0].bar(range(7), img, color=color)
    axs[0].set_title('Environment state')
    axs[0].set_ylabel('Metric value')


    axs[1].set_ylabel('Attribution value')
    x_data = np.arange(len(feature_names))
    plt.xticks(x_data, feature_names, rotation=25)
    attr = np.transpose(np.array(attributions), (1, 0, 2, 3))
    for i, ig_attr in enumerate(attributions):
        axs[i+1].set_title(f'Attributions for action: {action_names[i]}')
        axs[i+1].set_ylim([attr[idx].min(), attr[idx].max()])
        color = []
        ig_attr = ig_attr[idx][0]
        for val in ig_attr:
            if val < 0.0:
                color.append(NEGATIVE_VALUE_COLOR)
            else:
                color.append(POSITIVE_VALUE_COLOR)
        axs[i+1].bar(x_data, ig_attr, color=color)

def plot_attribution(idx, attributions, action, img, action_names=Utils.ACTION_NAMES, feature_names=Utils.FEATURE_NAMES):
    fig, axs = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
    color = []
    for val in img:
        if val < 0.0:
            color.append(NEGATIVE_VALUE_COLOR)
        else:
            color.append(POSITIVE_VALUE_COLOR)
    # plt.xticks(range(7), feature_names, rotation=30)
    axs[0].bar(range(7), img, color=color)
    axs[0].set_title('Environment state')
    axs[0].set_ylabel('Metric value')

    axs[1].set_ylabel('Attribution value')
    x_data = np.arange(len(feature_names))
    plt.xticks(x_data, feature_names, rotation=25)
    axs[1].set_ylim([attributions[idx][0].min(), attributions[idx][0].max()])
    axs[1].set_title(f'Attributions for action: {action_names[action]}')
    # plt.set_title(f'Attributions for action: {action_names[i]}')
    color = []
    ig_attr = attributions[idx][0]
    for val in ig_attr:
        if val < 0.0:
            color.append(NEGATIVE_VALUE_COLOR)
        else:
            color.append(POSITIVE_VALUE_COLOR)
    axs[1].bar(x_data, ig_attr, color=color)

def plot_attributions_cnn(idx, attributions, action, action_names=Utils.ACTION_NAMES, feature_names=Utils.FEATURE_NAMES, policy='mlp', data=None):
    frames = []
    for i, ig_attr in enumerate(attributions):
        if len(attributions) > 1:
            print(f'Attributions for action: {action_names[i]}')
        img = data[idx][0]
        # print(img.min(), img.max())
        img_ = np.transpose(img.cpu().detach().numpy(), (1, 2, 0))
        attr_ig_ = np.transpose(ig_attr[idx], (1, 2, 0))
        if attr_ig_.sum() == 0:
            print('Sum of attributions equals 0')
            continue
        
        fig_size = (10, 10)
        plt_fig, plt_axis = visualize_image_attr_multiple(attr=attr_ig_,
                                            original_image=img_,
                                            methods=["original_image", "blended_heat_map", "blended_heat_map"],
                                            signs=["all", "positive", "negative"],
                                            show_colorbar=True,
                                            titles=['Input', 
                                            f'IG positive attributions for action: {action_names[i]}', 
                                            f'IG negative attributions for action: {action_names[i]}'],
                                            fig_size=fig_size,
                                            use_pyplot=True,
                                            # plt_fig_axis=plt.subplots(figsize=fig_size)
                                            )

def plot_attribution_cnn(idx, attributions, action, action_names=Utils.ACTION_NAMES, feature_names=Utils.FEATURE_NAMES, policy='mlp', data=None):
    
    img = data[idx][0]
    # print(len(attributions))
    # print(img.min(), img.max())
    img_ = np.transpose(img.cpu().detach().numpy(), (1, 2, 0))
    attr_ig = np.transpose(attributions[idx], (1, 2, 0))
    # attr_ig_ = attr_ig_ + 0.01
    # attr_ig_ = np.clip(attr_ig_, -1, 1)
    if attr_ig.sum() == 0:
        print('Sum of attributions equals 0')
        return

    fig_size = (13, 7)
    plt_fig, plt_axis = visualize_image_attr_multiple(attr=attr_ig,
                                        original_image=img_,
                                        methods=["original_image", "blended_heat_map", "blended_heat_map"],
                                        signs=["all", "positive", "negative"],
                                        show_colorbar=True,
                                            titles=['Input', 
                                            f'IG positive attributions for action: {action_names[action]}', 
                                            f'IG negative attributions for action: {action_names[action]}'],
                                        fig_size=fig_size,
                                        use_pyplot=True,
                                        # plt_fig_axis=plt.subplots(figsize=fig_size)
                                        )

def plot_training_data(path, data_type, title=None):
    df = pd.read_csv(path, header=0, index_col=0)
    if data_type == 'reward':
        plt.figure(figsize=(20, 4))
        plt.plot(df['reward'])
        plt.title(f'Rewards: {title}')
        plt.xticks(range(len(df['reward'])))
        plt.show()
    elif data_type == 'episode_len':
        if 'episode_len' in df.columns:
            plt.figure(figsize=(20, 4))
            plt.plot(df['episode_len'])
            plt.title(f'Episode length: {title}')
            plt.show()

def plot_action_histogram(predictions, title=None):
    fig = plt.figure(figsize=(13, 2))
    plt.xticks(range(len(Utils.ACTION_NAMES)), Utils.ACTION_NAMES.values(), rotation=30)
    plt.ylabel('Count')
    plt.xlabel('Action')
    if title:
        plt.title(title)
    _ = plt.hist(predictions, bins=np.arange(0, len(Utils.ACTION_NAMES), .5), align='left')