package com.ryu.sonyremote.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

private val LightColors = lightColorScheme(
    primary = Color(0xFF176B5B),
    onPrimary = Color.White,
    primaryContainer = Color(0xFFC7EBDD),
    onPrimaryContainer = Color(0xFF092F28),
    secondary = Color(0xFFA2432F),
    onSecondary = Color.White,
    secondaryContainer = Color(0xFFFFDAD1),
    onSecondaryContainer = Color(0xFF3D0A02),
    tertiary = Color(0xFF4F5D75),
    background = Color(0xFFF4F4F2),
    surface = Color(0xFFFAFAF8),
    surfaceVariant = Color(0xFFE4E5E1),
    outline = Color(0xFF747875),
    error = Color(0xFFB3261E),
)

private val DarkColors = darkColorScheme(
    primary = Color(0xFF79D5BF),
    onPrimary = Color(0xFF00382F),
    primaryContainer = Color(0xFF075044),
    onPrimaryContainer = Color(0xFFA0F2DC),
    secondary = Color(0xFFFFAD98),
    onSecondary = Color(0xFF601707),
    secondaryContainer = Color(0xFF822B19),
    onSecondaryContainer = Color(0xFFFFDAD1),
    tertiary = Color(0xFFBEC7DC),
    background = Color(0xFF141615),
    surface = Color(0xFF191C1A),
    surfaceVariant = Color(0xFF404542),
    outline = Color(0xFF8A918D),
)

private val BaseTypography = Typography()
private fun TextStyle.withZeroSpacing(): TextStyle = copy(letterSpacing = 0.sp)

private val AppTypography = Typography(
    displayLarge = BaseTypography.displayLarge.withZeroSpacing(),
    displayMedium = BaseTypography.displayMedium.withZeroSpacing(),
    displaySmall = BaseTypography.displaySmall.withZeroSpacing(),
    headlineLarge = BaseTypography.headlineLarge.withZeroSpacing(),
    headlineMedium = BaseTypography.headlineMedium.withZeroSpacing(),
    headlineSmall = BaseTypography.headlineSmall.withZeroSpacing(),
    titleLarge = BaseTypography.titleLarge.withZeroSpacing(),
    titleMedium = BaseTypography.titleMedium.withZeroSpacing(),
    titleSmall = BaseTypography.titleSmall.withZeroSpacing(),
    bodyLarge = BaseTypography.bodyLarge.withZeroSpacing(),
    bodyMedium = BaseTypography.bodyMedium.withZeroSpacing(),
    bodySmall = BaseTypography.bodySmall.withZeroSpacing(),
    labelLarge = BaseTypography.labelLarge.withZeroSpacing(),
    labelMedium = BaseTypography.labelMedium.withZeroSpacing(),
    labelSmall = BaseTypography.labelSmall.withZeroSpacing(),
)

private val AppShapes = Shapes(
    extraSmall = RoundedCornerShape(4.dp),
    small = RoundedCornerShape(4.dp),
    medium = RoundedCornerShape(8.dp),
    large = RoundedCornerShape(8.dp),
    extraLarge = RoundedCornerShape(8.dp),
)

@Composable
fun AlphaCaptureLabTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = if (isSystemInDarkTheme()) DarkColors else LightColors,
        typography = AppTypography,
        shapes = AppShapes,
        content = content,
    )
}
