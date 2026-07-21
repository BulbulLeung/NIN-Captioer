interface Props {
  imagePath: string | null
  imageUrl: string | null
}

export function ImagePreview({ imagePath, imageUrl }: Props) {
  if (!imagePath || !imageUrl) {
    return (
      <div className="image-preview empty">
        <p>Select an image from the list to preview</p>
      </div>
    )
  }

  return (
    <div className="image-preview">
      <img src={imageUrl} alt="" />
    </div>
  )
}
