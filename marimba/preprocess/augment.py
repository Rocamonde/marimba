import avocado


def process_chunk(augmentor, chunk, verbose=True, **kwargs):
    # Load the reference dataset
    if verbose:
        print("Loading reference dataset...")
    dataset = avocado.load(
        kwargs['reference_dataset'],
        chunk=chunk,
        num_chunks=kwargs['num_chunks'],
    )

    # Augment the dataset
    if verbose:
        print("Augmenting the dataset...")
    augmented_dataset = augmentor.augment_dataset(
        kwargs['augmented_dataset'],
        dataset,
        kwargs['num_augments'],
    )

    # Save the augmented dataset
    if verbose:
        print("Saving the augmented dataset...")
    augmented_dataset.write()
